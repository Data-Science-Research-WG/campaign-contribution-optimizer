import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)


class CCO:
    """Campaign contribution optimizer algorithm implemented using dynamic programming technique."""

    def __init__(self, data: np.array):
        """
        :param: data, type numpy array
        """

        self.data = data
        self.n_channels = len(data[0, :])
        self.n_groups = len(data[:, 0])
        self.groups = np.array([i for i in range(0, self.n_groups)])
        self.allocation_revenue_matrix = None
        self.stage_channels_matrix = data[:, 0:2].transpose()
        self.output_max_positions = None
        self.output_max_per_group = None
        self.stage_n = 1

    def generate_allocation_matrix(self):
        """
        A method that will create the allocation matrix of no. of groups/discrete
        spend units between two channels.
        """
        stage_channels_matrix = self.stage_channels_matrix

        output_allocation_revenue_matrix = np.zeros([stage_channels_matrix[0, :].size,
                                                     stage_channels_matrix[0, :].size])

        for i in range(0, len(output_allocation_revenue_matrix[0, :])):
            for j in range(0, len(output_allocation_revenue_matrix[:, 0])):
                if i + j <= self.n_groups:
                    output_allocation_revenue_matrix[i, j] = stage_channels_matrix[0, i] + stage_channels_matrix[1, j]

        self.allocation_revenue_matrix = output_allocation_revenue_matrix

    def _get_value_position_combinations(self):
        """
        A method that returns the combination of values (expected revenues) as well as their respective positions where
        i + j = k, for some group k and channel 1 index i, channel 2 index j. This captures the number of possible
        combinations per no. of units to be allocated.
        :return: tuple, (combination of values per group, combination of positions per group)
        """
        groups = self.groups
        per_group_return_combinations = []
        combination_position = []
        for i in range(0, self.n_groups):
            per_group_return_combinations.append([])
            combination_position.append([])

        if not isinstance(self.allocation_revenue_matrix, type(None)):
            allocation_revenue_matrix = self.allocation_revenue_matrix
        else:
            self.generate_allocation_matrix()
            allocation_revenue_matrix = self.allocation_revenue_matrix

        for i in range(0, len(allocation_revenue_matrix[0])):
            for j in range(0, len(allocation_revenue_matrix[1])):
                for k in groups:
                    if i + j == k:
                        per_group_return_combinations[k].append(allocation_revenue_matrix[j, i])
                        combination_position[k].append([j, i])

        return per_group_return_combinations, combination_position

    def get_per_group_max_argmax(self):

        per_group_revenue_combinations, combination_positions = self._get_value_position_combinations()

        if isinstance(self.output_max_positions, type(None)):
            output_max_per_group = []
            argmax_per_group = []
            for i in range(0, len(per_group_revenue_combinations)):
                # get the max values per group.
                output_max_per_group.append(np.array(per_group_revenue_combinations[i]).max())
                # get the max argument position.
                y = np.array(per_group_revenue_combinations[i]).argmax()  # step 1 - we know x, and we're getting y.
                argmax_per_group.append(y)

            output_max_positions = []
            for i, combination in enumerate(combination_positions):
                output_max_positions.append(combination[argmax_per_group[i]])

            self.output_max_positions = output_max_positions
            self.output_max_per_group = output_max_per_group

        else:
            output_max_per_group = []
            argmax_per_group = []
            for i in range(0, len(per_group_revenue_combinations)):
                # get the max values per group.
                output_max_per_group.append(np.array(per_group_revenue_combinations[i]).max())
                # get the max argument position.
                y = np.array(per_group_revenue_combinations[i]).argmax()  # step 1 - we know x, and we're getting y.
                argmax_per_group.append(y)

            output_max_positions = []
            # Step 3 of algorithm to record the positions of (z, 0) for each z.
            for i, y in enumerate(argmax_per_group):
                max_combination = self.output_max_positions[i - y] + [y]
                output_max_positions.append(max_combination)

            # store the new output combination
            self.output_max_per_group = output_max_per_group
            self.output_max_positions = output_max_positions

    def next_stage_channels_matrix(self):
        if self.stage_n < (self.n_channels - 1):
            self.stage_n = self.stage_n + 1
            next_stage_data = self.data[:, self.stage_n].transpose()
            self.stage_channels_matrix = np.array([self.output_max_per_group, next_stage_data])
