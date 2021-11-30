from hls4ml.model.optimizer import OptimizerPass

class MergeRelu(OptimizerPass):
    def match(self, node):
        is_match = node.__class__.__name__ == 'Activation' and \
            node.get_input_node().__class__.__name__ in ['Conv1D', 'Conv2D', 'Dense']
        return is_match

    def transform(self, model, node):
        #Merge ReLU and Convolution layer if needed
        if not node.get_output_nodes():
            print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
            model.remove_node(node, rewire=False)
        else:
            model.remove_node(node, rewire=True)
        return True