//-----------------------------------------------------------
// ann_gpu_init() with ANN_GPU set: CTest runs this with ANN_GPU naming a
// backend no build has, and with "none", and checks what it prints (the
// message must say the backend is missing and list the build's own).
//-----------------------------------------------------------

#include <stdio.h>
#include "ann.h"

int main(void)
{
	// returns 0: no GPU backend started
	return ann_gpu_init() == 0 ? 0 : 1;
}
