#version 330

uniform sampler2D adjTex;
uniform float currentColumn;
uniform float maxColumns;
uniform float columnStride;

in vec2 in_Indices;

flat out vec2 vs_out_Indices;
out float vs_out_alphaTime;

void main()
{
	vec3 Pos=texture(adjTex,in_Indices.xy).xyz;
	gl_Position=vec4(Pos,1);

	vs_out_Indices=in_Indices;


	////////////////////////////ALPHATIME////////////////////////////////////////
	if(in_Indices.y+columnStride*2>currentColumn)
		vs_out_alphaTime=1-in_Indices.y+currentColumn;
	else
		vs_out_alphaTime=currentColumn-in_Indices.y;
}