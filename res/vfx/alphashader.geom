
#version 330
#define ROOT3 1.7320508075689

uniform float shapeStrength;

smooth in float[] vs_out_alphaTime;
smooth in float[] vs_out_alphaCurvature;
smooth in vec3[] vs_out_normal;

out float[] gs_out_alphaTime;//not 'flat' because of interpolation between the time of diffrent vertices for a smoother transparency in the fragemnt shader
out float[] gs_out_alphaCurvature;
out vec3[] gs_out_normal;
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

	gs_out_normal[0]=vs_out_normal[0];
	gs_out_normal[1]=vs_out_normal[1];
	gs_out_normal[2]=vs_out_normal[2];

	gs_out_alphaTime[0]=vs_out_alphaTime[0];//durchreichen zum fragmentshader
	gs_out_alphaTime[1]=vs_out_alphaTime[1];
	gs_out_alphaTime[2]=vs_out_alphaTime[2];

	gs_out_alphaCurvature[0]=vs_out_alphaCurvature[0];//durchreichen zum fragmentshader
	gs_out_alphaCurvature[1]=vs_out_alphaCurvature[1];
	gs_out_alphaCurvature[2]=vs_out_alphaCurvature[2];

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
}