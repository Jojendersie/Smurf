
#version 330
#define maxv 3

layout(triangles) in;
layout(triangle_strip, max_vertices = maxv) out;

#define ROOT3 1.7320508075689

uniform mat4 ProjectionView;
uniform vec3 eyePos;
uniform float b;
uniform float shapeStrength;
uniform float maxColumns;
uniform float maxRows;
uniform sampler2D adjTex;

in vec2[] vs_out_Indices;
in float[] vs_out_alphaTime;

out vec4 gs_out_worldPos;
out vec3 gs_out_normal;
out float gs_out_alphaTime;
out float gs_out_alphaCurvature;
out float gs_out_alphaShape;
out float gs_out_area;

void main()
{	
	vec3 adj[6];
	vec2 indices[3];
	vec2 indexStride;
	indexStride.x=1.0/maxRows;
	indexStride.y=1.0/maxColumns;
	for(int l=0;l<maxv;l++)
	{
		//////////////////////NORMAL/////////////////////////////////////////////
		gs_out_normal=normalize(cross(gl_in[0].gl_Position.xyz,gl_in[1].gl_Position.xyz));
		if(dot(gl_in[0].gl_Position.xyz-eyePos,gs_out_normal)<0)
			gs_out_normal*=-1;

		////////////////////////AREA////////////////////////////
		float lengths[3],s;
		lengths[0]=length(gl_in[0].gl_Position.xyz);
		lengths[1]=length(gl_in[1].gl_Position.xyz);
		lengths[2]=length(gl_in[2].gl_Position.xyz);

		s=(lengths[0]+lengths[1]+lengths[2])*0.5;

		gs_out_area=sqrt(s*(s-lengths[0])*(s-lengths[1])*(s-lengths[2]));

		///////////////ALPHASHAPE/////////////////////////////////
		vec3 d;
		d.x=length(gl_in[2].gl_Position.xyz-gl_in[1].gl_Position.xyz);
		d.y=length(gl_in[0].gl_Position.xyz-gl_in[2].gl_Position.xyz);
		d.z=length(gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz);
		float dmax=max(d.x*d.y,max(d.y*d.z,d.z*d.x));

		gs_out_alphaShape=clamp(pow((4.0*gs_out_area)/(ROOT3*dmax),shapeStrength),0.0,1.0);

		/////////////////////////////ALPHATIME////////////////////////////////////////
		gs_out_alphaTime=vs_out_alphaTime[l];

		///////////////////////////ALPHACURVATURE//////////////////////////////////////
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

		float etmp=0;
		int j=0;
		for(int i=0;i<6;i++)
		{
			vec3 tmp=adj[i]-gl_in[l].gl_Position.xyz;
			etmp=max(etmp,abs(dot(gs_out_normal,normalize(tmp))));
		}

		gs_out_alphaCurvature==clamp(1.0-b*etmp,0.0,1.0);

		/////////////////////////////POSITIONS////////////////////////////////////////
		gs_out_worldPos=gl_in[l].gl_Position;;
		//gs_out_worldPos[l].xyz/=gs_out_worldPos[l].w;
		gl_Position=ProjectionView*gl_in[l].gl_Position;

		EmitVertex();
	}
	EndPrimitive();
}