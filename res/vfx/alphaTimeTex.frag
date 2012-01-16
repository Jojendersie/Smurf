
 uniform sampler2D timeTex;
 uniform vec2 textureInfo;//is elapsed Time and TextureWidth

 varying float vertIndex;

void main()
{
	ivec2 index;
	index.x=mod(vertIndex,textureInfo.y);
	index.y=floor(vertIndex/textureInfo.y);

	gl_Color=texelFetch(timeTex,index,0).r+textureInfo.x;
}