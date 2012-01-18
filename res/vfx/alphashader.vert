#version 330

uniform float b;
uniform mat4 ProjectionView;
uniform sampler2D adjTex;
uniform float gridWidth;

in vec3 in_Pos;
in vec3 in_normal;
in float in_time;
in float vertexID;
/*
in vec3 in_adj1;
in vec3 in_adj2;
in vec3 in_adj3;
in vec3 in_adj4;
in vec3 in_adj5;
*/

out float vs_out_alphaCurvature;
out vec3 vs_out_normal;
out float vs_out_alphaTime;

void main()
{
	gl_Position=ProjectionView*vec4(in_Pos,1);
	vs_out_normal=in_normal;//normale durchreichen

	ivec2 index;
	index.x=int(mod(vertexID,gridWidth));
	index.y=int(floor(vertexID/gridWidth));

	vs_out_alphaTime=in_time;

	float etmp=0,e=0;
	int j=0;
	vec3 adj[6];
	for(int i=0;i<6;i++)
	{
		adj[i]=texelFetch(adjTex,index,0).xyz;
		vec3 tmp=adj[i]-in_Pos;//index must be changed to access the right adjacent vertex on the vertex map
		etmp=abs(dot(in_normal,tmp/dot(tmp,tmp)));//precalculation saves 5 square roots for one extra calculation, i bet it's worth it
		if(e<etmp)
		{
			e=etmp;
			j=i;
		}
	}

	vec3 tmp=adj[j]-in_Pos;
	e=abs(dot(in_normal,tmp/sqrt(dot(tmp,tmp))));

	vs_out_alphaCurvature=clamp(1.0-b*e,0.0,1.0);
}

