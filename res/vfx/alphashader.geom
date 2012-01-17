
#version 330
#define OUT_VERTS 3
#define ROOT3 1.7320508075689

layout(triangles) in;
layout(triangle,max_vertices = OUT_VERTS) out;


uniform float shapeStrength;

smooth in float[] in_alphaTime;
smooth in float[] in_alphaCurvature;
smooth in vec3 in_normal;

out float[] out_alphaTime;//not 'flat' because of interpolation between the time of diffrent vertices for a smoother transparency in the fragemnt shader
out float[] out_alphaCurvature;
out vec3 out_normal;
out float alphaShape;
out float area;

void main()
{
	out_normal=in_normal;

	out_alphaTime[0]=in_alphaTime[0];//durchreichen zum fragmentshader
	out_alphaTime[1]=in_alphaTime[1];
	out_alphaTime[2]=in_alphaTime[2];

	out_alphaCurvature[0]=in_alphaCurvature[0];//durchreichen zum fragmentshader
	out_alphaCurvature[1]=in_alphaCurvature[1];
	out_alphaCurvature[2]=in_alphaCurvature[2];

	float l[3],s;
	l[0]=sqrt(dot(gl_in[0].gl_Position,gl_in[0].gl_Position));
	l[1]=sqrt(dot(gl_in[1].gl_Position,gl_in[1].gl_Position));
	l[2]=sqrt(dot(gl_in[2].gl_Position,gl_in[2].gl_Position));

	s=(l[0]+l[1]+l[2])*0.5;

	float area=sqrt(s*(s-l[0])*(s-l[1])*(s-l[2]));

	vec3 vtmp;
	float d[3];

	vtmp=gl_in[2].gl_Position-gl_in[1].gl_Position;
	d[0]=dot(vtmp,vtmp);
	
	vtmp=gl_in[0].gl_Position-gl_in[2].gl_Position;
	d[1]=dot(vtmp,vtmp);

	vtmp=gl_in[1].gl_Position-gl_in[0].gl_Position;
	d[2]=dot(vtmp,vtmp);

	float dmax=sqrt(max(d[0],max(d[1],d[2])));

	alphaShape=clamp(pow((4.0*area)/(ROOT3*dmax),shapeStrength));
}