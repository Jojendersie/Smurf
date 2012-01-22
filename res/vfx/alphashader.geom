
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

#define ROOT3 1.7320508075689

uniform float b;
uniform float shapeStrength;
uniform float maxColumns;
uniform float maxRows;
uniform sampler2D adjTex;

in vec2[] vs_out_Indices;
in float[] vs_out_alphaTime;

out vec3 gs_out_normal;
out float[] gs_out_alphaTime;//not 'flat' because of interpolation between the time of diffrent vertices for a smoother transparency in the fragemnt shader
out float gs_out_alphaCurvature[3];
flat out float gs_out_alphaShape;
flat out float gs_out_area;

void main()
{
	gl_Position=gl_in[0].gl_Position;
	EmitVertex();
	gl_Position=gl_in[1].gl_Position;
	EmitVertex();
	gl_Position=gl_in[2].gl_Position;
	EmitVertex();
	EndPrimitive();

	gs_out_normal=cross(gl_in[0].gl_Position.xyz,gl_in[1].gl_Position.xyz);

	gs_out_alphaTime[0]=vs_out_alphaTime[0];//durchreichen zum fragmentshader
	gs_out_alphaTime[1]=vs_out_alphaTime[1];
	gs_out_alphaTime[2]=vs_out_alphaTime[2];

	float l[3],s;
	l[0]=sqrt(dot(gl_in[0].gl_Position,gl_in[0].gl_Position));
	l[1]=sqrt(dot(gl_in[1].gl_Position,gl_in[1].gl_Position));
	l[2]=sqrt(dot(gl_in[2].gl_Position,gl_in[2].gl_Position));

	s=(l[0]+l[1]+l[2])*0.5;

	float gs_out_area=sqrt(s*(s-l[0])*(s-l[1])*(s-l[2]));

	vec3 vtmp;
	float d[3];
	vtmp=gl_in[2].gl_Position.xyz-gl_in[1].gl_Position.xyz;
	d[0]=dot(vtmp,vtmp);
	
	vtmp=gl_in[0].gl_Position.xyz-gl_in[2].gl_Position.xyz;
	d[1]=dot(vtmp,vtmp);

	vtmp=gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz;
	d[2]=dot(vtmp,vtmp);

	float dmax=sqrt(max(d[0],max(d[1],d[2])));

	gs_out_alphaShape=clamp(pow((4.0*gs_out_area)/(ROOT3*dmax),shapeStrength),0.0,1.0);

	vec3 adj[6];
	vec2 indices[3];
	vec2 indexStride;
	indexStride.x=1.0/maxRows;
	indexStride.y=1.0/maxColumns;
	for(int l=0;l<3;l++)
	{
		indices[l]=vs_out_Indices[l];

		indices[l].x-=indexStride.x;
		adj[0]=texture(adjTex,indices[l]).xyz;

		indices[l].y-=indexStride.y;
		adj[1]=texture(adjTex,indices[l]).xyz;

		indices[l].x=vs_out_Indices[l].x;
		adj[2]=texture(adjTex,indices[l]).xyz;

		indices[l].y=vs_out_Indices[l].y;
		indices[l].x+=indexStride.x;
		adj[3]=texture(adjTex,indices[l]).xyz;

		indices[l].y+=indexStride.y;
		indices[l].x+=indexStride.x;
		adj[4]=texture(adjTex,indices[l]).xyz;

		indices[l].x=vs_out_Indices[l].x;
		adj[5]=texture(adjTex,indices[l]).xyz;

		float etmp=0,e=0;
		int j=0;
		for(int i=0;i<6;i++)
		{
			vec3 tmp=adj[i]-gl_in[l].gl_Position.xyz;//index must be changed to access the right adjacent vertex on the vertex map
			etmp=abs(dot(gs_out_normal,tmp/dot(tmp,tmp)));//precalculation saves 5 square roots for one extra calculation, i bet it's worth it
			if(e<etmp)
			{
				e=etmp;
				j=i;
			}
		}

		vec3 tmp=adj[j]-gl_in[l].gl_Position.xyz;
		gs_out_alphaCurvature[l]=clamp(1.0-b*abs(dot(gs_out_normal,tmp/sqrt(dot(tmp,tmp)))),0.0,1.0);
	}
}