 
 uniform float gridWidth;
 uniform float gridHeight;
 
 in vec3 in_Pos;
 in float in_vertIndex;

 varying vec3 vs_out_vertPos;

 void main()
 {
 	ivec2 index;
	index.x=mod(in_vertIndex,gridWidth);
	index.y=floor(in_vertIndex/gridWidth);
	gl_Position=vec4(float(index.x)/gridWidth*2.0-1.0,float(index.y)/gridHeight*2.0-1.0,0,1);

	vs_out_vertPos=in_Pos;
 }