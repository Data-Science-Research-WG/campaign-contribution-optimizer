import pandas as pd
import numpy as np
from CCO_dyna_prog import CCO

df = pd.read_csv("input_data.csv", header=0)
data = np.array(df[['Influencer Campaign Revenue Simulation',
                    'Magazine Page Spread Revenue Simulation',
                    'Digital Inspiration Campaign Revenue Simulation',
                    'Digital Product Launch Campaign']])

cco = CCO(data)
allocation_matrix = None

for i in range(0, cco.n_channels - 1):
    cco.generate_allocation_matrix()
    cco.get_per_group_max_argmax()
    cco.next_stage_channels_matrix()

spend_distribution = np.array(cco.output_max_positions) * (df['Spend'][1] - df['Spend'][0])
final_allocation_df = pd.DataFrame(spend_distribution,
                                   columns=['Influencer Campaign Revenue Simulation',
                                            'Magazine Page Spread Revenue Simulation',
                                            'Digital Inspiration Campaign Revenue Simulation',
                                            'Digital Product Launch Campaign'],
                                   index=df['Spend'])

final_allocation_df['Max Simulated Revenue'] = cco.output_max_per_group
breakpoint()
final_allocation_df.to_csv("output.csv")
