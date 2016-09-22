import matplotlib.pyplot as plt
import networkx as nx


class TFIDF_to_Graph():
    def init(self, TFIDF_model):
        self.model = TFIDF_model
        self.matrix = TFIDF_model.matrix
        self.words = TFIDF_model.words
        self.basic_path = TFIDF_model.basic_path
        self.option = TFIDF_model.option


    def plot_the_Graph(self,graph,filename):
        nx.draw(graph)
        write_path=self.basic_path[5]
        plt.save_fig(write_path+filename)


    def first_build_Graph(self):
        G = nx.Graph()
        return G


    def sub_matrix_to_node_edge(self,sub_mat):
        col_list = sub_mat.nonzero()[1]
        edge_list = [(x,y) for x in col_list for y in col_list if x!=y]
        return col_list, edge_list


class WeightedGraph(TFIDF_to_Graph):
    def weightFunction(self):
        

    def edge_to_weight(self,edge_list):


    def update_Graph(self):



    def build_Graph(self):
        G = self.first_build_Graph()





