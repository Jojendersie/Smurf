
#version 330
#define maxv 3

layout(triangles) in;
layout(triangle_strip, max_vertices = maxv) out;

#define ROOT3 1.7320508075689
#define FOUR_DIV_ROOT3 2.309401077

uniform mat4 ProjectionView;
uniform vec3 eyePos;
uniform float b;
uniform float shapeStrength;
uniform float rowStride;
uniform float columnStride;
uniform sampler2D adjTex;

flat in vec2[] vs_out_Indices;
in float[] vs_out_alphaTime;

out vec4 gs_out_worldPos;
flat out vec3 gs_out_normal;
out float gs_out_alphaTime;
out float gs_out_alphaCurvature;
out float gs_out_alphaShape;
out float gs_out_area;

void main()
{	
	vec3 adj[6];
	vec2 indices[3];
	for(int l=0;l<maxv;l++)
	{
		//////////////////////NORMAL/////////////////////////////////////////////
		gs_out_normal=cross(gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz,gl_in[2].gl_Position.xyz-gl_in[0].gl_Position.xyz);
		if(dot(gl_in[0].gl_Position.xyz-eyePos,gs_out_normal)<=0)
			gs_out_normal*=-1;

		////////////////////////AREA////////////////////////////
		float s;
		vec3 d;
		d.x=length(gl_in[2].gl_Position.xyz-gl_in[1].gl_Position.xyz);
		d.y=length(gl_in[0].gl_Position.xyz-gl_in[2].gl_Position.xyz);
		d.z=length(gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz);

		s=(d.x+d.y+d.z)*0.5;

		gs_out_area=sqrt(s*(s-d.x)*(s-d.y)*(s-d.z));

		///////////////ALPHASHAPE/////////////////////////////////
		float dmax=max(d.x*d.y,max(d.y*d.z,d.z*d.x));

		gs_out_alphaShape=clamp(pow((FOUR_DIV_ROOT3*gs_out_area)/dmax,shapeStrength),0.0,1.0);

		/////////////////////////////ALPHATIME////////////////////////////////////////
		gs_out_alphaTime=vs_out_alphaTime[l];

		///////////////////////////ALPHACURVATURE//////////////////////////////////////
		indices[l]=vs_out_Indices[l];

		indices[l].x-=rowStride;
		adj[0]=texture(adjTex,indices[l]).xyz;

		indices[l].y+=columnStride;
		adj[1]=texture(adjTex,indices[l]).xyz;

		indices[l].x=vs_out_Indices[l].x;
		adj[2]=texture(adjTex,indices[l]).xyz;

		indices[l].y=vs_out_Indices[l].y;
		indices[l].x+=rowStride;
		adj[3]=texture(adjTex,indices[l]).xyz;

		indices[l].y-=columnStride;
		indices[l].x+=rowStride;
		adj[4]=texture(adjTex,indices[l]).xyz;

		indices[l].x=vs_out_Indices[l].x;
		adj[5]=texture(adjTex,indices[l]).xyz;

		float etmp=0;
		int j=0;
		for(int i=0;i<6;i++)
		{
			vec3 tmp=adj[i]-gl_in[l].gl_Position.xyz;
			etmp=max(etmp,abs(dot(normalize(gs_out_normal),normalize(tmp))));
		}

		gs_out_alphaCurvature=clamp(1.0-b*etmp,0.0,1.0);

		/////////////////////////////POSITIONS////////////////////////////////////////
		gs_out_worldPos=gl_in[l].gl_Position;
		//gs_out_worldPos[l].xyz/=gs_out_worldPos[l].w;
		gl_Position=ProjectionView*gl_in[l].gl_Position;

		EmitVertex();
	}
	EndPrimitive();
}