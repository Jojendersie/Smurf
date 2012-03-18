#version 330

#define FILTERSIZE 15

////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////
uniform sampler2D opaqueSampler;

uniform sampler2D texSampler0;
uniform sampler2D texSampler1;
uniform sampler2D texSampler2;
uniform sampler2D texSampler3;
uniform sampler2D texSampler4;
uniform sampler2D texSampler5;
uniform sampler2D texSampler6;
uniform sampler2D texSampler7;

uniform float renderLayer;

out vec4 out_Color;

in vec2 out_vs_texCoords;


////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void main()
{	
	vec4 opaqueColor=out_Color=texture(opaqueSampler,out_vs_texCoords).rgba;
	
	vec4 Color=texture(texSampler0,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.5),0,1));

	if(renderLayer==1)
		return;

	Color=texture(texSampler1,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.33333333),0,1));

	if(renderLayer==2)
		return;

	Color=texture(texSampler2,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.25),0,1));

	if(renderLayer==3)
		return;

	Color=texture(texSampler3,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.2),0,1));

	if(renderLayer==4)
		return;

	Color=texture(texSampler4,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.166666666),0,1));

	if(renderLayer==5)
		return;

	Color=texture(texSampler5,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.142857142),0,1));

	if(renderLayer==6)
		return;

	Color=texture(texSampler6,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.125),0,1));

	if(renderLayer==7)
		return;

	Color=texture(texSampler7,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*Color.rgb,clamp(out_Color.a+(Color.a*0.11111111111),0,1));
	

	//DEPTH-PEELING-TEST
	/*
	vec4 Color=texture(texSampler0,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(1,0,0),clamp(out_Color.a+(Color.a*0.5),0,1));

	if(renderLayer==1)
		return;

	Color=texture(texSampler1,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(0,1,0),clamp(out_Color.a+(Color.a*0.33333333),0,1));

	if(renderLayer==2)
		return;

	Color=texture(texSampler2,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(0,0,1),clamp(out_Color.a+(Color.a*0.25),0,1));

	if(renderLayer==3)
		return;

	Color=texture(texSampler3,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(1,1,0),clamp(out_Color.a+(Color.a*0.2),0,1));

	if(renderLayer==4)
		return;

	Color=texture(texSampler4,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(0,1,1),clamp(out_Color.a+(Color.a*0.166666666),0,1));

	if(renderLayer==5)
		return;

	Color=texture(texSampler5,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(1,0,1),clamp(out_Color.a+(Color.a*0.142857142),0,1));

	if(renderLayer==6)
		return;

	Color=texture(texSampler6,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(1,1,1),clamp(out_Color.a+(Color.a*0.125),0,1));

	if(renderLayer==7)
		return;

	Color=texture(texSampler7,out_vs_texCoords).rgba;
	out_Color=vec4((1-Color.a)*out_Color.rgb + Color.a*vec3(0,0,0),clamp(out_Color.a+(Color.a*0.11111111111),0,1));
	*/
}
