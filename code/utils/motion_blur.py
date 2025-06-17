# ----- VN START -----

import torch
from kornia.filters import MotionBlur  # or from your library

def random_motion_blur(x, 
                       kmin:int=3, kmax:int=15,    # odd kernel sizes between 3 and 15
                       amin:float=0.0, amax:float=360.0,
                       dmin:float=-1.0, dmax:float=1.0):
    # pick a random odd kernel size
    k = torch.randint(kmin//2, kmax//2 + 1, (1,)).item() * 2 + 1  
    # random angle in [0,360)
    angle = torch.rand(1, device=x.device) * (amax - amin) + amin
    # random direction in [-1,1]
    direction = torch.rand(1, device=x.device) * (dmax - dmin) + dmin

    mb = MotionBlur(
        kernel_size=k,
        angle=angle,
        direction=direction
    )
    return mb(x)

# ----- VN END -----