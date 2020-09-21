# %%
import time

from data_generation import generate_data
from data_dominance import generate_data_dominance

# %%
if __name__ == "__main__":
    # %%
    start_time = time.time()
    data_fname = generate_data()
    dominance_data_fname = generate_data_dominance(data_fname)
    print(time.time() - start_time)
