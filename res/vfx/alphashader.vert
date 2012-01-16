#version 330

uniform float b;
uniform mat4 ProjectionView;
uniform sampler2D timeTex;
uniform float texWidth;

attribute vec3 in_Pos;
attribute vec3 in_O_normal;
attribute vec3 in_adj[6];
attribute float vertexID;
/*
in vec3 in_adj1;
in vec3 in_adj2;
in vec3 in_adj3;
in vec3 in_adj4;
in vec3 in_adj5;
*/

out float alphaCurvature;
out vec3 in_normal;
out float in_alphaTime;

void main()
{
	gl_Position=ProjectionView*vec4(in_Pos,1);
	in_normal=in_O_normal;//normale durchreichen

	vec2 index;
	index.x=mod(vertexID,texWidth);
	index.y=floor(vertexID/texWidth);
	in_alphaTime=texelFetch(timeTex,index).r;

	float j=0,etmp=0,e=0;
	for(int i=0;i<6;i++)
	{
		vec3 tmp=in_adj[i]-in_Pos;
		etmp=abs(dot(in_O_normal,tmp/dot(tmp,tmp)));//precalculation saves 5 square roots for one extra calculation, i bet it's worth it
		if(e<etmp)
		{
			e=etmp;
			j=i;
		}
	}

	vec3 tmp=in_adj[j]-in_Pos;
	e=abs(dot(in_O_normal,tmp/sqrt(dot(tmp,tmp))));

	alphaCurvature=clamp(1.0-b*e);
}

