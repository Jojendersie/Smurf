
////////////////////////////////////////////////////////////////////////////////
// Paramter Declaration and Definiton
////////////////////////////////////////////////////////////////////////////////

//char* pcFile = "res\\data\\BubbleChamber_11x11x10_T0.am";
//char* pcFile = "res\\data\\SquareCylinder_192x64x48_T3048.am";
//char* pcFile = "res\\data\\SquareCylinder\\flow_t####.am";
char* pcFile = "res\\data\\Wing_128x64x32_T0.am";
bool SMOKE_TIME_DEPENDENT_INTEGRATION=false;
unsigned int SMOKE_TIME_STEPSIZE=40;//stepsize in ms, every SMOKE_TIME_STEPSIZE ms a new slice is being taken
unsigned int SMOKE_PARTICLE_NUMBER=5000;
float SMOKE_PRISM_THICKNESS=0.1f;
float SMOKE_DENSITY_CONSTANT=0.5f;
float SMOKE_DENSITY_CONSTANT_K;//initialized in main.cpp
unsigned short RENDER_SMURF_ROWS=50;
unsigned short RENDER_SMURF_COLUMS=250;
float SMOKE_AREA_CONSTANT_NORMALIZATION=.01f;
float SMOKE_SHAPE_CONSTANT=1.0f;
float SMOKE_CURVATURE_CONSTANT=2.0f;
float SMOKE_COLOR[]={0.5f,0.5f,0.5f};

unsigned short PROGRAM_FRAMES_PER_RELEASE = 3;
unsigned short PROGRAM_NUM_SEEDLINES = 1;//Number of Seedlines in [1,9]
unsigned short RENDER_DEPTH_PEELING_LAYER = 8;//Peeling layer from 1 up to 8
float RENDER_SMURF_STEPSIZE = 0.25f;
unsigned int RENDER_POLYGON_MAX = 75000;