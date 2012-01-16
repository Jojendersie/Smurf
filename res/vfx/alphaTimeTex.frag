
 uniform sampler2D timeTex;
 uniform vec2 textureInfo;//is elapsed Time and TextureWidth

 in float vertIndex;

void main()
{
	vec2 index;
	index.x=mod(vertIndex,textureInfo.y);
	index.y=floor(vertIndex/textureInfo.y);

	gl_Color=texelFetch(timeTex,index).r+textureInfo.x;
}