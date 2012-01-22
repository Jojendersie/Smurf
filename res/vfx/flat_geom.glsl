
#version 330
#define maxv 3

layout(triangles) in;
layout(triangle_strip, max_vertices = maxv) out;

uniform mat4 ProjectionView;

in vec3 vs_out_worldPos[3];

out vec3 gs_out_worldPos;
out vec3 gs_out_normal;

void main()
{
	for(int i=0;i<maxv;i++)
	{
		gs_out_normal=normalize(cross(gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz,gl_in[2].gl_Position.xyz-gl_in[0].gl_Position.xyz));
		gs_out_worldPos=vs_out_worldPos[i];
		gl_Position=gl_in[i].gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}