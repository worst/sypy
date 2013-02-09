#    SyPy: A Python framework for evaluating graph-based Sybil detection
#    algorithms in social and information networks.
#
#    Copyright (C) 2013  Yazan Boshmaf
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import utils
import networkx as nx
import random
import math
import copy

from results import *


class BaseDetector:

    def __init__(self, network):
        self.network = network
        self.__check_integrity()

        self.honests_truth = self.network.left_region.graph.nodes()
        self.honests_predicted = None

    def __check_integrity(self):
        if not self.network.is_stitched:
            raise Exception("Network is not stitched")

    def detect(self):
        raise NotImplementedError("This method is not supported")

    def _vote_honests_predicted(self, collection):
        self.honests_predicted = []
        biggest_overlap = 0
        for i, list_item in enumerate(collection):
            overlap = set.intersection(
                set(self.network.known_honests),
                set(list_item)
            )
            if len(overlap) > biggest_overlap:
                biggest_overlap = len(overlap)
                self.honests_predicted = list_item

class GenericBCCDetector(BaseDetector):

    def __init__(self, network):
        BaseDetector.__init__(self, network)

    def detect(self):
        bcc = nx.biconnected_components(self.network.graph.structure)

        self._vote_honests_predicted(bcc)
        return Results(self)

class LouvainCommunityDetector(BaseDetector):
    def __init__(self, network, resolution=0.000001):
        BaseDetector.__init__(self, network)
        self.resolution = resolution
        self.max_iter = 1
        self.dendrogram = []
        self.passes = 0

    def detect(self):
        structure = self.network.graph.structure.copy()
        self.__construct_dendrogram(structure)

        results = self.dendrogram[0]
        for dendro in range(1, len(self.dendrogram)):
            for node in range(structure.order()):
                results[node] = self.dendrogram[dendro][results[node]]

        reverse = {}
        for key in results.keys():
            value = results[key]
            list = reverse.get(value, [])
            list.append(key)
            reverse[value] = list

        self._vote_honests_predicted(reverse.values())
        return Results(self)

    def __construct_dendrogram(self, structure):
        self.passes += 1
        print self.passes, structure.order(), structure.size(weight='weight')
        # Initialize the attributes
        structure.graph['communities'] = []
        structure.graph['size'] = structure.size(weight='weight')
        structure.graph['weights_within'] = {}
        structure.graph['weights_incident'] = {}
        for node in structure:
            community = structure.subgraph(node)
            structure.node[node]['community'] = community
            structure.graph['communities'].append(community)
            community.graph['weights_within'][community] = 0
            community.graph['weights_incident'][community] =\
                structure.degree(node, weight='weight')
            structure.node[node]['weights_incident'] =\
                community.graph['weights_incident'][community]

        # Phase 1
        outer_mod_increase, iter_count = 0, 0
        while True:
            inner_mod_increase = 0

            for node in structure:
                max_mod_increase = 0
                best_node = None
                print node, structure.neighbors(node)
                for neighbor in structure.neighbors(node):
                    mod_increase = self.__modularity(
                        structure,
                        node,
                        neighbor
                    )
                    if mod_increase > max_mod_increase:
                        max_mod_increase = mod_increase
                        best_node = neighbor
                if best_node != None:
                    self.__move_node(
                        structure,
                        node,
                        best_node
                    )
                inner_mod_increase += max_mod_increase
            outer_mod_increase += inner_mod_increase
            iter_count += 1
            if inner_mod_increase < self.resolution or\
                iter_count > self.max_iter:
                print inner_mod_increase
                break

        # Phase 2
        community_graph = nx.Graph()
        community_graph.add_nodes_from(
            range(len(structure.graph['communities']))
        )

        community_dict = {}
        count = 0
        for community in structure.graph['communities']:
            community_dict[community] = count
            count += 1

        node_dict = {}
        for node in structure:
            node_dict[node] = community_dict[structure.node[node]['community']]
        self.dendrogram.append(node_dict)

        for (u, v) in structure.edges():
            u_node = community_dict[structure.node[u]['community']]
            v_node = community_dict[structure.node[v]['community']]
            try:
                weight = community_graph[v_node][u_node]['weight'] +\
                    structure[u][v].get('weight', 1)
            except KeyError:
                weight = structure[u][v].get('weight', 1)
            community_graph.add_edge(u_node, v_node, weight=weight)

        if outer_mod_increase <= self.resolution:
            return

        self.__construct_dendrogram(community_graph)

    def __modularity(self, structure, node, neighbor):
        if node == neighbor:
            return 0

        community = structure.node[neighbor]['community']
        weights_between = reduce(lambda x, y: x+y,
            [structure[node][neighbor].get('weight', 1)\
            for neighbor in structure.neighbors(node)\
            if neighbor in community])

        mod_change  = (community.graph['weights_within'][community] +\
            weights_between) / (2 * structure.graph['size']) -\
            ((community.graph['weights_incident'][community] +\
            structure.node[node]['weights_incident']) /\
            (2 * structure.graph['size'])) ** 2 -\
            (community.graph['weights_within'][community]) /\
            (2 * structure.graph['size']) -\
            (community.graph['weights_incident'][community] /\
            (2 * structure.graph['size'])) ** 2 -\
            (structure.node[node]['weights_incident'] /\
            (2 * structure.graph['size'])) ** 2

        return  mod_change

    def __move_node(self, structure, source, target):
        source_comm = structure.node[source]['community']
        source_comm.remove_node(source)

        target_comm = structure.node[target]['community']
        structure.node[source]['community'] = target_comm
        target_comm.add_node(source)

        empty = False
        if source_comm.order() == 0:
            structure.graph['communities'].remove(source_comm)
            del source_comm
            empty = True

        target_comm.graph['weights_within'][target_comm] =\
            target_comm.size(weight='weight')

        weights_between = reduce(lambda x, y: x+y,
            [structure[source][neighbor].get('weight', 1)\
            for neighbor in structure.neighbors(source)\
            if neighbor in target_comm])

        target_comm.graph['weights_incident'][target_comm] +=\
            structure.node[source]['weights_incident'] - weights_between

        if not empty:
            source_comm.graph['weights_within'][source_comm] =\
                source_comm.size(weight='weight')

            source_comm.graph['weights_incident'][source_comm] -=\
                structure.node[source]['weights_incident'] + weights_between


class GirvanNewmanCommunityDetector(BaseDetector):
    """
    Implements Girvan-Newman community detection algorithm as described
    in Community Structure in Social and Biological Networks, Girvan et al.
    PNAS June, Vol 99(12), 2002.

    Note: The algorithm performs a top-down, hierarchical graph clustering
    based on edge betweenness. It tries to partition the network into two
    tightly-knit communities or clusters, as a way to detect Sybils.
    """
    def __init__(self, network, max_level=1):
        BaseDetector.__init__(self, network)
        self.dendogram = nx.DiGraph()
        self.max_level = max_level

    def detect(self):
        structure = self.network.graph.structure.copy()

        self.dendogram.add_node(structure)
        self.__construct_dendogram(structure, 1)

        sub_structures = self.dendogram.nodes()
        sub_structures.remove(structure)

        communities = []
        for sub_structure in sub_structures:
            communities.append(sub_structure.nodes())

        self._vote_honests_predicted(communities)
        return Results(self)

    def __construct_dendogram(self, structure, current_level):
        if structure.order() <= 1 or current_level > self.max_level:
            return

        edge_betweenness = nx.edge_betweenness_centrality(
            structure,
            normalized=False
        )

        max_edge = max(
            edge_betweenness,
            key=edge_betweenness.get
        )

        (left, right) = (0, 1)
        structure.remove_edge(
            max_edge[left],
            max_edge[right]
        )

        sub_structures = nx.connected_component_subgraphs(structure)
        if len(sub_structures) == 1:
            self.__construct_dendogram(
                structure,
                current_level
            )
        else:
            self.__add_dendogram_level(
                structure,
                sub_structures
            )
            self.__construct_dendogram(
                sub_structures[left],
                current_level+1
            )
            self.__construct_dendogram(
                sub_structures[right],
                current_level+1
            )

    def __add_dendogram_level(self, structure, sub_structures):
        (left, right) = (0, 1)
        self.dendogram.add_node(sub_structures[left])
        self.dendogram.add_node(sub_structures[right])

        self.dendogram.add_edge(
            structure,
            sub_structures[left]
        )
        self.dendogram.add_edge(
            structure,
            sub_structures[right]
        )


class BaseSybilDetector(BaseDetector):

    def __init__(self, network, verifiers, seed):
        BaseDetector.__init__(self, network)

        self.verifiers = verifiers
        self.seed = seed

        self.__check_integrity()

    def __check_integrity(self):
        if self.seed:
            random.seed(seed)

        if not self.verifiers:
            self.verifiers = random.sample(
                self.honests_truth,
                random.randint(1, len(self.network.known_honests))
            )

        valid_verifiers = set(self.verifiers).issubset(set(self.honests_truth))
        if not valid_verifiers:
            raise Exception("Invalid verifiers. Not subset of honests")


class SybilGuardDetector(BaseSybilDetector):
    """
    Implements a centralized version of the SybilGuard protocol as described
    in SybilGuard: Defending Against Sybil Attacks via Social Networks,
    Yu et al., SIGCOMM (2006).

    Note: In this centralized version, a set of verifiers which are a subset
    of the known honest nodes try to label nodes either honest or Sybil, and
    then the labeling that results in the correct inclusion of most known
    honests in the honest region is selected, instead of relying on a single
    verifier. Moreover, the number of honest nodes in the network is directly
    computed but not sampled as presented in the paper. This means that the
    implementation uses the exact value scaled by its asymptotic constant,
    but not its estimate.
    """
    def __init__(self, network, verifiers=None, route_len_scaler=1.0, seed=None):
        BaseSybilDetector.__init__(self, network, verifiers, seed)
        self.route_len_scaler = route_len_scaler

    def detect(self):
        self.__generate_random_routes()

        num_honests = len(self.honests_truth)
        route_len = int(
            self.route_len_scaler * math.sqrt(num_honests) *\
                math.log10(num_honests)
        )
        walks = self.__walk_random_routes(route_len)

        verified_honests = self.__accept_honests_from_verifiers(walks)
        self._vote_honests_predicted(verified_honests)

        return Results(self)

    def __generate_random_routes(self):
        nodes = self.network.graph.nodes()
        random_routes = {}

        for node in nodes:
            node_routes = {}
            neighbors = self.network.graph.structure.neighbors(node)
            shuffled_neighbors = copy.copy(neighbors)
            random.shuffle(shuffled_neighbors)
            for index, neighbor in enumerate(neighbors):
                node_routes[neighbor] = shuffled_neighbors[index]

            node_routes[node] = random.choice(neighbors)
            random_routes[node] = node_routes

        nx.set_node_attributes(
            self.network.graph.structure,
            "random_routes",
            random_routes
        )

    def __walk_random_routes(self, route_len):
        walks = {}
        structure = self.network.graph.structure

        for node in structure.nodes():
            walk = [node]

            ingress_node = node
            routing_node = node

            node_routes = structure.node[routing_node]["random_routes"]
            outgress_node = node_routes[ingress_node]
            while len(walk) != (route_len + 1):
                walk.append(outgress_node)
                routing_node = outgress_node

                node_routes = structure.node[routing_node]["random_routes"]
                outgress_node = node_routes[ingress_node]

                ingress_node = routing_node

            walks[node] = walk

        return walks

    def __accept_honests_from_verifiers(self, walks):
        verified_honests = []
        for verifier in self.verifiers:
            verifier_honests = []
            verifier_walk = self.__get_walk_edges(walks[verifier])
            for suspect in walks:
                suspect_walk = self.__get_walk_edges(walks[suspect])
                overlap = set.intersection(
                    set(verifier_walk),
                    set(suspect_walk)
                )
                if len(overlap) != 0:
                    verifier_honests.append(suspect)
            verified_honests.append(verifier_honests)

        return verified_honests

    def __get_walk_edges(self, walk):
        edges = []
        for index in xrange(len(walk)-1):
            edges.append(
                (walk[index], walk[index+1])
            )
        return edges


class SybilLimitDetector(BaseSybilDetector):
    """
    Implements a centralized version of the SybilLimit protocol, as described
    in SybilLimit: A Near-Optimial Social Network Defense against Sybil Attacks,
    Yu et al., IEEE S&P (2008).
    """
    def __init__(self, network, verifiers=None, route_len_scaler=1.0,
        num_instances_scaler=1.0, tail_balance_scalar=4.0, seed=None
    ):
        BaseSybilDetector.__init__(self, network, verifiers, seed)

        self.route_len_scaler = route_len_scaler
        self.num_instances_scaler = num_instances_scaler
        self.tail_balance_scalar = tail_balance_scalar

    def detect(self):
        num_edges = self.network.left_region.graph.size()
        num_instances = int(
            self.num_instances_scaler * math.sqrt(num_edges)
        )

        (lower_mtime, upper_mtime) = utils.compute_mixing_time_bounds(
            self.network.left_region.graph
        )
        route_len = int(
            self.route_len_scaler * math.ceil(upper_mtime)
        )

        self.__generate_secure_random_routes(num_instances)
        suspects_tails = self.__walk_secure_random_routes(
            route_len,
            num_instances,
            verify=False
        )

        self.__generate_secure_random_routes(num_instances)
        verifiers_tails = self.__walk_secure_random_routes(
            route_len,
            num_instances
        )

        verified_honests = self.__accept_honests_from_verifiers(
            suspects_tails,
            verifiers_tails,
            num_instances
        )
        self._vote_honests_predicted(verified_honests)

        return Results(self)

    def __generate_secure_random_routes(self, num_instances):
        nodes = self.network.graph.nodes()
        secure_routes = {}

        for node in nodes:
            route_instances = []
            while len(route_instances) != num_instances:
                node_routes = {}
                neighbors = self.network.graph.structure.neighbors(node)
                shuffled_neighbors = copy.copy(neighbors)
                random.shuffle(shuffled_neighbors)
                for index, neighbor in enumerate(neighbors):
                    node_routes[neighbor] = shuffled_neighbors[index]
                node_routes[node] = random.choice(neighbors)
                route_instances.append(node_routes)

            secure_routes[node] = route_instances

        nx.set_node_attributes(
            self.network.graph.structure,
            "secure_routes",
            secure_routes
        )

    def __walk_secure_random_routes(self, route_len, num_instances, verify=True):
        tails = {}
        structure = self.network.graph.structure

        nodes = self.verifiers
        if not verify:
            nodes = list(
                set(structure.nodes()) - set(nodes)
            )

        for node in nodes:
            instance_tails = []
            for instance_index in xrange(num_instances):
                walk = [node]

                ingress_node = node
                routing_node = node

                route_instances = structure.node[routing_node]["secure_routes"]
                node_routes = route_instances[instance_index]

                outgress_node = node_routes[ingress_node]
                while len(walk) != (route_len + 1):
                    walk.append(outgress_node)
                    routing_node = outgress_node

                    route_instances = structure.node[routing_node]["secure_routes"]
                    node_routes = route_instances[instance_index]
                    outgress_node = node_routes[ingress_node]

                    ingress_node = routing_node

                instance_tails.append(walk[-2:])

            tails[node] = instance_tails

        return tails

    def __accept_honests_from_verifiers(self, suspects_tails, verifiers_tails,
        num_instances
    ):
        verified_honests = []
        for verifier in verifiers_tails:
            verifier_honests = []
            tail_counters = [0] * num_instances
            for suspect in suspects_tails:
                overlap = self.__find_tail_intersections(
                    suspects_tails[suspect],
                    verifiers_tails[verifier]
                )
                (accepted, tail_counters) = self.__update_tail_counters(
                    verifiers_tails[verifier],
                    tail_counters,
                    overlap,
                    num_instances
                )
                if accepted:
                    verifier_honests.append(suspect)
            verified_honests.append(verifier_honests)

        return verified_honests

    def __find_tail_intersections(self, suspect_tails, verifier_tails):
        overlap = []
        for tail in suspect_tails:
            if tail in verifier_tails:
                index = verifier_tails.index(tail)
                overlap.append([tail, index])

        return overlap

    def __update_tail_counters(self, verifier_tails, tail_counters,
        overlap, num_instances
    ):
        accepted = False
        if not overlap:
            return (accepted, tail_counters)

        average_load = (1.0 + sum(tail_counters))/(float)(num_instances)

        threshold = self.tail_balance_scalar * max(
            math.log10(num_instances),
            average_load
        )

        relevant_counters = []
        indexes = []
        for (tail, index) in overlap:
            relevant_counters.append(tail_counters[index])
            indexes.append(index)

        min_index = indexes[relevant_counters.index((min(relevant_counters)))]

        if not ((tail_counters[min_index] + 1.0) > threshold):
            accepted = True
            tail_counters[min_index] += 1

        return (accepted, tail_counters)


