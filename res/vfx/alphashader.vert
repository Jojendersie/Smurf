#version 330

uniform sampler2D adjTex;
uniform float currentColumn;

in vec2 in_Indices;

flat out vec2 vs_out_Indices;
out float vs_out_alphaTime;

void main()
{
	vec3 Pos=texture(adjTex,in_Indices.xy).xyz;
	gl_Position=vec4(Pos,1);

	vs_out_Indices=in_Indices;


	////////////////////////////ALPHATIME////////////////////////////////////////
	if(in_Indices.y>currentColumn)
		vs_out_alphaTime=in_Indices.y-currentColumn;//maxColumns-in_Indices.y+currentColumn+1/maxColumns;
	else
		vs_out_alphaTime=currentColumn-in_Indices.y;
}