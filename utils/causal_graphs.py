def get_dsprites_causal_graph():
    return {
        0: [],      # shape root
        1: [],      # scale root
        2: [0],     # rotation depends on shape
        3: [0, 1],  # posX depends on shape + scale
        4: [0, 1],  # posY depends on shape + scale
    }

def get_identity_graph(latent_dim):
    return {i: [] for i in range(latent_dim)}
