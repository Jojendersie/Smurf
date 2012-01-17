 uniform vec2 textureInfo;

 attribute float vertIndex;
 attribute float startTime;

 out float time;

 void main()
 {
	time=startTime;
	vec2 index;
	index.x=mod(vertIndex,textureInfo.y);
	index.y=floor(vertIndex/textureInfo.y);
	
	gl_Position=vec4((index.x*textureInfo.x)*2.0-1.0,(index.y*textureInfo.x)*2.0-1.0,0.0,1.0);
 }