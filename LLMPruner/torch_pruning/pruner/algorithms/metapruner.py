import numpy as np
import torch
import torch.nn as nn
import typing
import matplotlib.pyplot as plt

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency


class MetaPruner:
    """
        Meta Pruner for structural pruning.

        Args:
            model (nn.Module): A to-be-pruned model
            example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            importance (Callable): importance estimator.
            global_pruning (bool): enable global pruning. 
            ch_sparsity (float): global channel sparisty.
            ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
            iterative_steps (int): number of steps for iterative pruning.
            iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            ignored_layers (List[nn.Module]): ignored modules.

            round_to (int): channel rounding.
            customized_pruners (dict): a dict containing module-pruner pairs.
            unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.
            root_module_types (list): types of prunable modules.
            output_transform (Callable): A function to transform network outputs.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        global_pruning: bool = False,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None,

        # Advanced
        round_to: int = None,  # round channels to 8x, 16x, ...
        # for grouped channels.
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # for consecutive channels.
        consecutive_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        # unwrapped nn.Parameters like ViT.pos_emb
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        root_module_types: typing.List = [
            ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        root_instances: typing.List = None,
        forward_fn: typing.Callable = None,
        output_transform: typing.Callable = None,
        enable_index_mapping: bool = False,
        # imp_hm_path: str = 'importance_heatmap_default'
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning

        self.channel_groups = channel_groups
        self.consecutive_groups = consecutive_groups
        self.root_module_types = root_module_types
        self.root_instances = root_instances
        self.round_to = round_to

        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
        )

        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The customized channel sparsity for different layers
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # detect group convs & group norms
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups
        
        if self.global_pruning: # TODO: Support both ch_groups and consecutive_groups in a single forward
            initial_total_channels = 0
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )

                if ch_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // ch_groups)
                elif consecutive_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // consecutive_groups)
                else:
                    initial_total_channels += self.DG.get_out_channels(group[0][0].target.module) 
                
            self.initial_total_channels = initial_total_channels
        
        if enable_index_mapping:
            for node in self.DG.module2node.values():
                node.enable_index_mapping = True

        self.groups_importance_attn = None
        self.groups_importance_mlp = None

        # self.imp_hm_path = imp_hm_path
    
    def pruning_history(self):
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history):
        self.DG.load_pruning_history(pruning_history)

    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
            self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass

    def step(self, interactive=False):
        self.current_step += 1
        if self.global_pruning:
            if interactive:
                return self.prune_global()
            else:
                for group in self.prune_global():
                    group.prune()
        else:
            if interactive:
                return self.prune_local()
            else:
                for group in self.prune_local():
                    # pass
                    # print(group)
                    group.prune()

    def estimate_importance(self, group, ch_groups=1, consecutive_groups=1):
        return self.importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping
    
    def get_consecutive_groups(self, group):
        if isinstance(self.consecutive_groups, int):
            return self.consecutive_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.consecutive_groups:
                return self.consecutive_groups[module]
        return 1  # no channel grouping

    def prune_local(self):
        if self.current_step > self.iterative_steps:
            return

        dummy_grp_num = 1
        self.groups_importance_attn = []
        self.groups_importance_mlp = []

        # print("MAX-CH-SPARSITY", self.max_ch_sparsity)
        # print("LEN-ALL-GROUP", len(self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances)))
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if imp is None: continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                    
                if n_pruned <= 0:
                    continue

                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]

                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)

                # TEST
                # print(dummy_grp_num, "IMPORTANCE TEST", imp, len(imp))
                is_attention = 'empty'
                if len(imp) <= 32:
                    self.groups_importance_attn.append(imp)
                    is_attention = 'attention'
                else:
                    self.groups_importance_mlp.append(imp)
                    is_attention = 'MLP'



                imp_argsort = torch.argsort(imp)
                
                if ch_groups > 1:
                    pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
                    group_size = current_channels//ch_groups
                    pruning_idxs = torch.cat(
                        [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                elif consecutive_groups > 1:
                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                        for i in pruning_groups], 0)
                else:
                    # print("NO-CHCG")
                    pruning_idxs = imp_argsort[:n_pruned]

                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())


                #print(dummy_grp_num, is_attention, len(pruning_idxs))
                # #print(dummy_grp_num, "GROUP IMPORTANCE TEST", group)
                dummy_grp_num += 1

                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self):
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if imp is None: continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)
                global_importance.append((group, ch_groups, consecutive_groups, imp))

        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        print(imp.shape, len(global_importance))
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        print(n_pruned, target_sparsity, self.initial_total_channels)
        # print("NNNNNNNN", n_pruned, target_sparsity, self.initial_total_channels)
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        
        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, consecutive_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            
            if pruning_indices.size(-1) == 0:
                continue
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if consecutive_groups > 1:
                group_size = consecutive_groups
                pruning_indices = torch.cat(
                    [torch.tensor([j+group_size*i for j in range(group_size)])
                    for i in pruning_indices], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                yield group

    def visualize_importance(self, imp_hm_path):
        # print("self.groups_importance_attn", self.groups_importance_attn)
        # if is_2nd:
        #     imp_matrix_mlp = torch.stack(self.groups_importance_mlp, dim=0)
        #     imp_matrix_mlp = imp_matrix_mlp.cpu().numpy()
        #     plt.imshow(imp_matrix_mlp, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=0.02)
        #
        #     plt.colorbar()
        #     plt.set_xlabel('Group')  # Label for x-axis
        #     plt.set_ylabel('Layer')
        #     plt.set_title('Importance MLP')
        #
        #     path = '/home/haohuiwu/LLM-Pruner/outputs/' + imp_hm_path + '.png'
        #     # Save the plot as a PNG file
        #     plt.savefig(path)
        #
        #     plt.close()
        #     return

        imp_matrix_attn = torch.stack(self.groups_importance_attn, dim=0)
        imp_matrix_attn = imp_matrix_attn.cpu().numpy()

        imp_matrix_mlp = torch.stack(self.groups_importance_mlp, dim=0)
        imp_matrix_mlp = imp_matrix_mlp.cpu().numpy()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

        # Create the first heatmap
        im1 = axs[0].imshow(imp_matrix_attn, cmap='hot', interpolation='nearest', aspect='auto')
        fig.colorbar(im1, ax=axs[0])
        axs[0].set_xlabel('Group')  # Label for x-axis
        axs[0].set_ylabel('Layer')

        upper_bound = np.percentile(imp_matrix_mlp, 99.9)

        # Create the second heatmap
        im2 = axs[1].imshow(imp_matrix_mlp, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=upper_bound)
        fig.colorbar(im2, ax=axs[1])
        axs[1].set_xlabel('Group')  # Label for x-axis
        axs[1].set_ylabel('Layer')

        # Set titles for the subplots
        axs[0].set_title('Importance Attention')
        axs[1].set_title('Importance MLP')

        path = '/home/haohuiwu/LLM-Pruner/outputs/' + imp_hm_path + '.png'
        # Save the plot as a PNG file
        plt.savefig(path)

        # Close the plot to free up memory
        plt.close()