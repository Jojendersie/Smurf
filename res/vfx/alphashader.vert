#version 330

uniform mat4 ProjectionView;
uniform sampler2D adjTex;
uniform float currentColumn;
uniform float maxColumns;

in vec3 in_Pos;
in vec2 in_Indices;

out vec2 vs_out_Indices;
out float vs_out_alphaTime;

void main()
{
	vec3 Pos=texture(adjTex,in_Indices).xyz;
	//Pos=in_Pos;
	gl_Position=ProjectionView*vec4(Pos,1);

	vs_out_Indices=in_Indices;

	if(in_Indices.y>currentColumn)
		vs_out_alphaTime=in_Indices.y-currentColumn;
	else
		vs_out_alphaTime=maxColumns-(currentColumn-in_Indices.y);
}

