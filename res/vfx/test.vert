#version 330

uniform sampler2D adjTex;
uniform mat4 ProjectionView;

in vec2 in_Indices;

void main()
{
	vec3 Pos=texture(adjTex,in_Indices.xy).xyz;
	gl_Position=ProjectionView*vec4(Pos,1);
}