from hls4ml.model.optimizer import OptimizerPass

class MergeRelu(OptimizerPass):
    def match(self, node):
        is_match = node.__class__.__name__ == 'Activation' and \
            (node.get_input_node().__class__.__name__ == 'Conv2D' or 
            node.get_input_node().__class__.__name__ == 'Conv2DBatchnorm')
        return is_match

    def transform(self, model, node):
        #Merge ReLU and Convolution layer if needed
        next_node = next((x for x in model.graph.values() if node.outputs[0] in x.inputs), None)
        next_next_node  = next((x for x in model.graph.values() if next_node.outputs[0] in x.inputs), None)
        # print("---------------------------------")
        # print("printing for " + node.get_input_node().__class__.__name__)
        # print("output is " + node.get_input_node().get_output_variable().name)
        # print("input is " + node.get_input_node().get_input_variable().name)
        # print("output type is " + node.get_input_node().get_output_variable().type.name)
        # print("input type is " + node.get_input_node().get_input_variable().type.name)
        # print("---------------------------------")
        # print("printing for " + node.__class__.__name__)
        # print("input is " + node.get_input_variable().name)
        # print("input type is " + node.get_input_variable().type.name)
        # print("output is " + node.get_output_variable().name)
        # print("output type is " + node.get_output_variable().type.name)
        # print("---------------------")
        # print("printing for next node : " + next_node.__class__.__name__)
        # print("input is " + next_node.get_input_variable().name)
        # print("input type is " + next_node.get_input_variable().type.name)
        # print("output is " + next_node.get_output_variable().name)
        # print("output type is " + next_node.get_output_variable().type.name)
        # print("---------------------")
        # print("printing for next next node: " + next_next_node.__class__.__name__)
        # print("input is " + next_next_node.get_input_variable().name)
        # print("input type is " + next_next_node.get_input_variable().type.name)
        # print("output is " + next_next_node.get_output_variable().name)
        # print("output type is " + next_next_node.get_output_variable().type.name)
        # print("---------------------")
        if not node.get_output_nodes():
            print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
            model.remove_node(node, rewire=False)
        else:
            model.remove_node(node, rewire=True)
        return True