from hls4ml.model.optimizer import OptimizerPass

class MergeRelu(OptimizerPass):
    def match(self, node):
        print("-------------------------------------------")
        print("layer name is ")
        print(self.__class__.__name__)
        print("-------------------------------------------")
        is_match = node.__class__.__name__ == 'Activation' and \
            (node.get_input_node().__class__.__name__ == 'Conv2D' or 
            node.get_input_node().__class__.__name__ == 'Conv2DBatchnorm')
        return is_match

    def transform(self, model, node):
        #Merge ReLU and Convolution layer if needed
        next_node = next((x for x in model.graph.values() if node.outputs[0] in x.inputs), None)
        print("---------------------")
        print(next_node.inputs)
        print(next_node.__class__.__name__)
        print(next_node.outputs)
        print("---------------------")
        if not node.get_output_nodes():
            print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
            model.remove_node(node, rewire=False)
        else:
            model.remove_node(node, rewire=True)
        return True