
#define kDielectricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04)
#define kDieletricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04)
#define FLT_EPS  5.960464478e-8
#define FLT_MIN  1.175494351e-38
#define FLT_MAX  3.402823466e+38
#define HALF_EPS 4.8828125e-4
#define HALF_MIN 6.103515625e-5
#define HALF_MIN_SQRT 0.0078125
#define HALF_MAX 65504.0
#define UINT_MAX 0xFFFFFFFFu
#define INT_MAX  0x7FFFFFFF

float4x4 UNITY_MATRIX_V : WORLDVIEW;
float4x4 UNITY_MATRIX_VP : WORLDVIEWPROJECTION;
float4x4 UNITY_MATRIX_I_V : WORLDVIEW;
float4x4 UNITY_MATRIX_P : PROJECTION;
float4x4 unity_ObjectToWorld : WORLD;
float4x4 unity_WorldToObject : WORLDINVERSE;

float3 _WorldSpaceLightPos : DIRECTION < string Object = "Light"; >;
float3 _WorldSpaceCameraPos : POSITION < string Object = "Camera"; >;
float3 LightAmbient : AMBIENT < string Object = "Light"; >;

struct G_VertexColor
{
    float4 OutLineColor;
    float OutLineWidth;
    float OutLineOffset;
    float RampAddID;
    float RimMask;
};

struct BRDFData
{
    half3 albedo;
    half3 diffuse;
    half3 specular;
    half reflectivity;
    half perceptualRoughness;
    half roughness;
    half roughness2;
    half grazingTerm;
    half normalizationTerm;
    half roughness2MinusOne;
};

#define _ALPHAPREMULTIPLY_ON
#define _ALPHATEST_ON
#define _USE_REFLECTION_SPHERE
#define _USE_EYE_REFLECTION_TEXTURE
#define _USE_REFLECTION_TEXTURE
#define _RAMPADD_ON
#define _LAYERMAP_ON
//#define IS_HAIRCOVER_PASS

float4 _BaseColor = float4(1, 1, 1, 1);
float4 _DefValue = float4(0.497999996, 0.720000029, 0, 0.194999993);
float4 _RampAddColor = float4(0.4, 0.403921, 0.43529, 1);
float _VertexColor = 1;
float4 _SpecularThreshold = float4(0.100000001, 0.5, 1, 1);
float4 _FadeParam = float4(0.75, 2, 0.400000006, 4);
float _ShaderType = 0;
float _ClipValue = 0.33;
float _LayerWeight = 0;
float _SkinSaturation = 1;
float4 _MultiplyColor = float4(1,1,1,1);
float4 _MultiplyOutlineColor = float4(1, 1, 1, 1);
float4 _BaseMap_ST = float4(1, 1, 0, 0);
float4 _MatCapParam = float4(0.300000012, 1, 1, 0);
float4 _MainLightParam = float4(0.340000004, 0.569999993, 0.74000001, 0);
float4 _MatCapLightColor = float4(1, 1, 1, 1);
float4 _ShadeMultiplyColor = float4(0.65098, 0.431372, 0.415686, 1);
float4 _ShadeAdditiveColor = float4(0, 0, 0, 0);
float4 _EyeHighlightColor = float4(1, 1, 1, 1);
float4 _VLSpecColor = float4(1, 1, 1, 1);
float4 _VLEyeSpecColor = float4(0.74901, 0.74901, 0.74901, 1);;
float4 _MatCapRimColor = float4(2.52894664, 8.47418976, 7.41788101, 1);
float4 _MatCapRimLight = float4(0.400000006, -0.25999999, 0.870000005, 15);
float4 _GlobalLightParameter = float4(0.25, 1, 0.00999999978, 1);
float4 _ReflectionSphereMap_HDR = float4(1, 1, 1, 1);
float4 _OutlineParam = float4(0.0500000007, 5, 0.100000001, 0.5);

bool IsOrtho;
float4 _MainLightShadowParams;
float4x4 _HeadMatrix : CONTROLOBJECT < string name = "model.pmx"; string item = "head"; >; //replace name to your model filename (Shift-JIS)
float3 _LocalFaceUp = float3(0, 1, 0);
float3 _LocalFaceRight = float3(1, 0, 0);
float3 _LocalFaceForward = float3(0, 0, -1);
float4x4 _MainLightWorldToShadow = float4x4(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
);

Texture2D _BaseMap : MATERIALTEXTURE;
Texture2D _ShadeMap < string ResourceName = "Texture2D/t_chr_fktn-cstm-0000_bdy_sdw.png"; >;
Texture2D _RampMap < string ResourceName = "Texture2D/t_chr_fktn-base-0000_rmp.png"; >;
Texture2D _HighlightMap;
Texture2D _DefMap < string ResourceName = "Texture2D/t_chr_fktn-cstm-0000_bdy_def.png"; >;
Texture2D _LayerMap;
Texture2D _RampAddMap < string ResourceName = "Texture2D/t_chr_fktn-cstm-0000_bdy_rma.png"; >;
Texture2D _ReflectionSphereMap;
TextureCube _VLSpecCube < string ResourceName = "skybox.png"; >;


half3 SampleSH(half3 normalWS)
{
    half3 ambientColor = LightAmbient;
    half influence = dot(normalWS, half3(0, 1, 0)) * 0.5 + 0.5;
    half3 fakeSH = ambientColor * influence;
    return max(fakeSH, half3(0, 0, 0));
}

float3 SafeNormalize(float3 inVec)
{
    float dp3 = max(FLT_MIN, dot(inVec, inVec));
    return inVec * rsqrt(dp3);
}

void Encode4BitTo8Bit(float4 A, float4 B, out float4 C)
{
    float4 HighBit = floor(A * 15.9375f + 0.03125f);
    float4 LowBit = floor(B * 15.9375f + 0.03125f);
    C = (HighBit * 16.0f + LowBit) / 255.0f;
}

void Decode4BitFrom8Bit(float4 C, out float4 A, out float4 B)
{
    const float k = 1.0f / 16.0f;
    float4 HighBit = floor(C * 15.9375f + 0.03125f);
    float4 LowBit = C * 255.0f - HighBit * 16.0f;
    A = HighBit * k;
    B = LowBit * k;
}

G_VertexColor DecodeVertexColor(float4 VertexColor)
{
    G_VertexColor OutColor;
    float4 LowBit, HighBit;
    Decode4BitFrom8Bit(VertexColor, HighBit, LowBit);
    OutColor.OutLineColor = float4(HighBit.x, LowBit.x, HighBit.y, LowBit.w);
    OutColor.OutLineWidth = LowBit.z;
    OutColor.OutLineOffset = HighBit.z;
    OutColor.RampAddID = LowBit.y;
    OutColor.RimMask = HighBit.w;
	
    return OutColor;
}

float PerceptualRoughnessToRoughness(float perceptualRoughness)
{
    return perceptualRoughness * perceptualRoughness;
}

float PerceptualSmoothnessToPerceptualRoughness(float perceptualSmoothness)
{
    return (1.0 - perceptualSmoothness);
}

half OneMinusReflectivityMetallic(half metallic)
{
    half oneMinusDielectricSpec = kDielectricSpec.a;
    return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
}

inline void InitializeBRDFDataDirect(half3 albedo, half3 diffuse, half3 specular, half reflectivity, half oneMinusReflectivity, half smoothness, inout half alpha, out BRDFData outBRDFData)
{
    outBRDFData = (BRDFData) 0;
    outBRDFData.albedo = albedo;
    outBRDFData.diffuse = diffuse;
    outBRDFData.specular = specular;
    outBRDFData.reflectivity = reflectivity;

    outBRDFData.perceptualRoughness = PerceptualSmoothnessToPerceptualRoughness(smoothness);
    outBRDFData.roughness = max(PerceptualRoughnessToRoughness(outBRDFData.perceptualRoughness), HALF_MIN_SQRT);
    outBRDFData.roughness2 = max(outBRDFData.roughness * outBRDFData.roughness, HALF_MIN);
    outBRDFData.grazingTerm = saturate(smoothness + reflectivity);
    outBRDFData.normalizationTerm = outBRDFData.roughness * half(4.0) + half(2.0);
    outBRDFData.roughness2MinusOne = outBRDFData.roughness2 - half(1.0);

#if defined(_ALPHAPREMULTIPLY_ON)
        outBRDFData.diffuse *= alpha;
#endif
}

inline void InitializeBRDFData(half3 albedo, half metallic, half3 specular, half smoothness, inout half alpha, out BRDFData outBRDFData)
{
    half oneMinusReflectivity = OneMinusReflectivityMetallic(metallic);
    half reflectivity = half(1.0) - oneMinusReflectivity;
    half3 brdfDiffuse = albedo * oneMinusReflectivity;
    half3 brdfSpecular = lerp(kDieletricSpec.rgb, albedo, metallic);
    InitializeBRDFDataDirect(albedo, brdfDiffuse, brdfSpecular, reflectivity, oneMinusReflectivity, smoothness, alpha, outBRDFData);
}

BRDFData G_InitialBRDFData(float3 BaseColor, float Smoothness, float Metallic, float Specular, bool IsEye)
{
    float OutAlpha = 1.0f;
    BRDFData G_BRDFData;
    InitializeBRDFData(BaseColor, Metallic, Specular, Smoothness, OutAlpha, G_BRDFData);
    G_BRDFData.grazingTerm = IsEye ? saturate(Smoothness + kDieletricSpec.x) : G_BRDFData.grazingTerm;
    G_BRDFData.diffuse = IsEye ? BaseColor * kDieletricSpec.a : G_BRDFData.diffuse;
    G_BRDFData.specular = IsEye ? BaseColor : G_BRDFData.specular;
	
    return G_BRDFData;
}


half G_DirectBRDFSpecular(BRDFData BrdfData, half3 NormalWS, half3 NormalMatS, float4 LightDir, float3 ViewDir)
{
    bool DisableMatCap = LightDir.w > 0.5f;
    ViewDir = DisableMatCap ? ViewDir : float3(0.0f, 0.0f, 1.0f);
    float3 HalfDir = SafeNormalize(LightDir.xyz + ViewDir);

    float3 Normal = DisableMatCap ? NormalWS : NormalMatS;
    float NoH = saturate(dot(float3(Normal), HalfDir));
    half LoH = half(saturate(dot(LightDir.xyz, HalfDir)));

    float D = NoH * NoH * BrdfData.roughness2MinusOne + 1.00001f;

    half LoH2 = LoH * LoH;
    half SpecularTerm = BrdfData.roughness2 / ((D * D) * max(0.1h, LoH2) * BrdfData.normalizationTerm);

    return SpecularTerm;
}

half3 EnvironmentBRDFSpecular(BRDFData brdfData, half fresnelTerm)
{
    float surfaceReduction = 1.0 / (brdfData.roughness2 + 1.0);
    return half3(surfaceReduction * lerp(brdfData.specular, brdfData.grazingTerm, fresnelTerm));
}

half DirectBRDFSpecular(BRDFData brdfData, half3 normalWS, half3 lightDirectionWS, half3 viewDirectionWS)
{
    float3 lightDirectionWSFloat3 = float3(lightDirectionWS);
    float3 halfDir = SafeNormalize(lightDirectionWSFloat3 + float3(viewDirectionWS));

    float NoH = saturate(dot(float3(normalWS), halfDir));
    half LoH = half(saturate(dot(lightDirectionWSFloat3, halfDir)));
    float d = NoH * NoH * brdfData.roughness2MinusOne + 1.00001f;

    half LoH2 = LoH * LoH;
    half specularTerm = brdfData.roughness2 / ((d * d) * max(0.1h, LoH2) * brdfData.normalizationTerm);
 
    return specularTerm;
}

float3 TransformObjectToWorld(float3 positionOS)
{
    return mul(unity_ObjectToWorld, float4(positionOS, 1.0)).xyz;
}

float3 TransformObjectToWorldNormal(float3 normalOS, bool doNormalize = true)
{
    // Normal need to be multiply by inverse transpose
    float3 normalWS = mul(normalOS, (float3x3) unity_WorldToObject);
    if (doNormalize)
        return SafeNormalize(normalWS);
    return normalWS;
}

float4 TransformWorldToShadowCoord(float3 positionWS)
{
#if defined(_MAIN_LIGHT_SHADOWS_SCREEN) && !defined(_SURFACE_TYPE_TRANSPARENT)
    float4 shadowCoord = float4(ComputeNormalizedDeviceCoordinatesWithZ(positionWS, GetWorldToHClipMatrix()), 1.0);
#else
#ifdef _MAIN_LIGHT_SHADOWS_CASCADE
        half cascadeIndex = ComputeCascadeIndex(positionWS);
#else
    half cascadeIndex = half(0.0);
#endif
    
    float4 shadowCoord = float4(mul(_MainLightWorldToShadow[cascadeIndex], float4(positionWS, 1.0)).xyz, 0.0);
#endif
    return shadowCoord;
}

half MainLightRealtimeShadow(float4 shadowCoord)
{
#if !defined(MAIN_LIGHT_CALCULATE_SHADOWS)
    return half(1.0);
#elif defined(_MAIN_LIGHT_SHADOWS_SCREEN) && !defined(_SURFACE_TYPE_TRANSPARENT)
        return SampleScreenSpaceShadowmap(shadowCoord);
#else
        ShadowSamplingData shadowSamplingData = GetMainLightShadowSamplingData();
        half4 shadowParams = GetMainLightShadowParams();
        return SampleShadowmap(TEXTURE2D_ARGS(_MainLightShadowmapTexture, sampler_LinearClampCompare), shadowCoord, shadowSamplingData, shadowParams, false);
#endif
}

float4 TransformWorldToHClip(float3 positionWS)
{
    return mul(UNITY_MATRIX_VP, float4(positionWS, 1.0));
}

float Pow4(float x)
{
    return (x * x) * (x * x);
}

float Luminance(float3 linearRgb)
{
    return dot(linearRgb, float3(0.2126729, 0.7151522, 0.0721750));
}

sampler sampler_BaseMap = sampler_state{texture = <_BaseMap>; MINFILTER = LINEAR;MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_ShadeMap = sampler_state{texture = <_ShadeMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_RampMap = sampler_state{texture = <_RampMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_HighlightMap = sampler_state{texture = <_HighlightMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_DefMap = sampler_state{texture = <_DefMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_LayerMap = sampler_state{texture = <_LayerMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_RampAddMap = sampler_state{texture = <_RampAddMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
sampler2D sampler_ReflectionSphereMap = sampler_state{texture = <_ReflectionSphereMap>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};
samplerCUBE sampler_VLSpecCube = sampler_state{texture = <_VLSpecCube>; MINFILTER = LINEAR; MAGFILTER = LINEAR; MIPFILTER = LINEAR;};

struct appdata
{
    float4 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Tangent : TANGENT;
    float2 UV0 : TEXCOORD0;
    float2 UV1 : TEXCOORD1;
    float4 Color : TEXCOORD3; //Vertex Color -->Additional UV3;
};

struct v2f
{
    float4 UV : TEXCOORD0;
    float3 PositionWS : TEXCOORD1;
    float4 Color1 : COLOR;
    float4 Color2 : TEXCOORD2;
    float3 NormalWS : TEXCOORD3;
    float3 NormalHeadReflect : TEXCOORD4;
    float4 ShadowCoord : TEXCOORD6;
};


v2f vert(appdata v)
{
    v2f o;

    o.UV.xy = v.UV0 * _BaseMap_ST.xy + _BaseMap_ST.zw;
    o.UV.zw = v.UV1.xy;
	
    o.PositionWS = TransformObjectToWorld(v.Position);
    o.NormalWS = TransformObjectToWorldNormal(v.Normal);
    float3 _HeadRightDirection = mul(_LocalFaceRight, ConvertTo3x3(_HeadMatrix));
    float3 _HeadUpDirection = mul(_LocalFaceUp, ConvertTo3x3(_HeadMatrix));
    float3 _HeadDirection = mul(_LocalFaceForward, ConvertTo3x3(_HeadMatrix));
    float4x4 _HeadXAxisReflectionMatrix = float4x4(
        _HeadRightDirection.x, _HeadUpDirection.x, _HeadDirection.x, 0.0f,
        _HeadRightDirection.y, _HeadUpDirection.y, _HeadDirection.y, 0.0f,
        _HeadRightDirection.z, _HeadUpDirection.z, _HeadDirection.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
    o.NormalHeadReflect = mul(_HeadXAxisReflectionMatrix, float4(v.Normal, 0.0f)).xyz;

    G_VertexColor VertexColor = DecodeVertexColor(v.Color);
    o.Color1 = VertexColor.OutLineColor;
    o.Color2 = float4(
		VertexColor.OutLineWidth,
		VertexColor.OutLineOffset,
		VertexColor.RampAddID,
		VertexColor.RimMask);
	
    o.ShadowCoord = TransformWorldToShadowCoord(o.PositionWS);
	
    float4 PositionWS = float4(o.PositionWS, 1.0f);
    return o;
}

float4 frag(v2f i, bool IsFront : SV_IsFrontFace) : SV_Target
{
#if defined(IS_HAIRCOVER_PASS) && !defined(_ENALBEHAIRCOVER_ON)
		clip(-1);
#endif
	
    G_VertexColor VertexColor;
    VertexColor.OutLineColor = i.Color1;
    VertexColor.OutLineWidth = i.Color2.x;
    VertexColor.OutLineOffset = i.Color2.y;
    VertexColor.RampAddID = i.Color2.z;
    VertexColor.RimMask = i.Color2.w;

    bool IsFace = _ShaderType == 9;
    bool IsHair = _ShaderType == 8;
    bool IsEye = _ShaderType == 4;
    bool IsEyeHightLight = _ShaderType == 5;
    bool IsEyeBrow = _ShaderType == 6;

    float3 NormalWS = normalize(i.NormalWS);
    NormalWS = IsFront ? NormalWS : NormalWS * -1.0f;
	
    float3 ViewVector = _WorldSpaceCameraPos - i.PositionWS;
    float3 ViewDirection = normalize(ViewVector);
    ViewDirection = IsOrtho ? UNITY_MATRIX_V[2].xyz : ViewDirection;

    float3 CameraUp = UNITY_MATRIX_V[1].xyz;
    float3 ViewSide = normalize(cross(ViewDirection, CameraUp));
    float3 ViewUp = normalize(cross(ViewSide, ViewDirection));
    float3x3 WorldToMatcap = float3x3(ViewSide, ViewUp, ViewDirection);

    float3 NormalMatS = mul(WorldToMatcap, float4(NormalWS, 0.0f));
    bool DisableMatCap = _MainLightParam.w > 0.5f;
    float4 _MatCapMainLight = DisableMatCap ? normalize(_MainLightParam) : _WorldSpaceLightPos;
    float NoL = dot(NormalWS, _MatCapMainLight);
    float MatCapNoL = dot(NormalMatS, _MatCapMainLight);
    NoL = DisableMatCap ? NoL : MatCapNoL;

    float Shadow = MainLightRealtimeShadow(i.ShadowCoord);
    float ShadowFadeOut = dot(-ViewVector, -ViewVector);
    ShadowFadeOut = saturate(ShadowFadeOut * _MainLightShadowParams.z + _MainLightShadowParams.w);
    ShadowFadeOut *= ShadowFadeOut;
    Shadow = lerp(Shadow, 1, ShadowFadeOut);
    Shadow = lerp(1.0f, Shadow, _MainLightShadowParams.x);
    Shadow = saturate(Shadow * ((4.0f * Shadow - 6) * Shadow + 3.0f));

    float3 LayerMapColor = 0;
    float LayerWeight = 0;
    float4 LayerMapDef = 0;
#ifdef _LAYERMAP_ON
    if (_LayerWeight != 0)
    {
        float2 LayerMapUV = i.UV * float2(0.5f, 1.0f);
        float4 LayerMap = tex2D(sampler_LayerMap, LayerMapUV);
        LayerMapColor = LayerMap.rgb;
        LayerWeight = LayerMap.a * _LayerWeight;
        
        float2 LayerMapDefUV = LayerMapUV + float2(0.5f, 0.0f);
        LayerMapDef = tex2D(sampler_LayerMap, LayerMapDefUV);
    }
#endif
	
    float4 BaseMap = tex2D(sampler_BaseMap, i.UV.xy);
#ifdef _LAYERMAP_ON
        BaseMap.rgb = lerp(BaseMap, LayerMapColor.rgb, LayerWeight);
#endif
    float4 ShadeMap = tex2D(sampler_ShadeMap, i.UV.xy);
    float4 DefMap = _DefValue;
#ifndef _DEFMAP_OFF
    DefMap = tex2D(sampler_DefMap, i.UV.xy).xyzw;
#endif
#ifdef _LAYERMAP_ON
        DefMap = lerp(DefMap, LayerMapDef, LayerWeight);
#endif
    float DefDiffuse = DefMap.x;
    float DefMetallic = DefMap.z;
    float DefSmoothness = DefMap.y;
    float DefSpecular = DefMap.w;

    float DiffuseOffset = DefDiffuse * 2.0f - 1.0f;
    float Smoothness = min(DefSmoothness, 1);
    float Metallic = IsFace ? 0 : DefMetallic;
	
    float SpecularIntensity = min(DefSpecular, Shadow);
    float3 NormalWorM = DisableMatCap ? NormalWS : NormalMatS;
    float3 ViewDirWorM = DisableMatCap ? ViewDirection : float3(0, 0, 1);

    if (IsHair)
    {
        float IsMicroHair = saturate(i.UV.x - 0.75f) * saturate(i.UV.y - 0.75f);
        IsMicroHair = IsMicroHair != 0;
	
        float HairSpecular = Pow4(saturate(dot(NormalWorM, ViewDirWorM)));
        HairSpecular = smoothstep(_SpecularThreshold.x - _SpecularThreshold.y, _SpecularThreshold.x + _SpecularThreshold.y, HairSpecular);
        HairSpecular *= SpecularIntensity;
        HairSpecular = IsMicroHair ? 0 : HairSpecular;
		
        float3 HighlightMap = tex2D(sampler_HighlightMap, i.UV.xy).xyz;
        BaseMap.xyz = lerp(BaseMap.xyz, HighlightMap.xyz, HairSpecular);
		

        float3 _HeadDirection = mul(_LocalFaceForward, ConvertTo3x3(_HeadMatrix));
        float HairFadeX = dot(_HeadDirection, ViewDirection);
        HairFadeX = _FadeParam.x - HairFadeX;
        HairFadeX = saturate(HairFadeX * _FadeParam.y);
        float3 _HeadUpDirection = mul(_LocalFaceUp, ConvertTo3x3(_HeadMatrix));
        float HairFadeZ = dot(_HeadUpDirection, ViewDirection);
        HairFadeZ = abs(HairFadeZ) - _FadeParam.z;
        HairFadeZ = saturate(HairFadeZ * _FadeParam.w);
	
        BaseMap.a = lerp(1, max(HairFadeX, HairFadeZ), BaseMap.a);
	
        SpecularIntensity *= IsMicroHair ? 1 : 0;
    }
	
    float4 RampAddMap = 0;
    float3 RampAddColor = 0;
#ifdef _RAMPADD_ON
    float2 RampAddMapUV = float2(saturate(DiffuseOffset + NormalMatS.z), VertexColor.RampAddID);
    RampAddMap = tex2D(sampler_RampAddMap, RampAddMapUV);
	RampAddColor = RampAddMap.xyz * _RampAddColor.xyz;
	
    float3 DiffuseRampAddColor = lerp(RampAddColor, 0, RampAddMap.a);
    BaseMap.xyz += DiffuseRampAddColor;
    ShadeMap.xyz += DiffuseRampAddColor;
#endif
	
    float BaseLighting = NoL * 0.5f + 0.5f;
    BaseLighting = saturate(BaseLighting + (DiffuseOffset - _MatCapParam.x) * 0.5f);

    float3 NormalHeadMatS = mul(WorldToMatcap, i.NormalHeadReflect.xyz);
    float FaceNoL = DisableMatCap ? dot(i.NormalHeadReflect, _MatCapMainLight) : dot(NormalHeadMatS, _MatCapMainLight);
    float FaceLighting = saturate((FaceNoL + DiffuseOffset) * 0.5f + 0.5f);
    FaceLighting = max(FaceLighting, BaseLighting);
    FaceLighting = lerp(BaseLighting, FaceLighting, DefMetallic);
	
    BaseLighting = IsFace ? FaceLighting : BaseLighting;
    BaseLighting = min(BaseLighting, Shadow);
	
    float2 RampMapUV = float2(BaseLighting, 0);
    float4 RampMap = tex2D(sampler_RampMap, RampMapUV);

    const float ShadowIntensity = _MatCapParam.z;
    float3 RampedLighting = lerp(BaseMap.xyz, ShadeMap.xyz * _ShadeMultiplyColor, RampMap.w * ShadowIntensity);
    float3 SkinRampedLighting = lerp(RampMap, RampMap.xyz * _ShadeMultiplyColor, RampMap.w);
    SkinRampedLighting = lerp(1, SkinRampedLighting, ShadowIntensity);
    SkinRampedLighting = BaseMap * SkinRampedLighting;
    RampedLighting = lerp(RampedLighting, SkinRampedLighting, ShadeMap.w);

    float SkinSaturation = _SkinSaturation - 1;
    SkinSaturation = SkinSaturation * ShadeMap.w + 1.0f;
    RampedLighting = lerp(Luminance(RampedLighting), RampedLighting, SkinSaturation);
    RampedLighting *= _BaseColor;
	
    RampedLighting = IsEyeHightLight ? RampedLighting * _EyeHighlightColor : RampedLighting;
    BRDFData G_BRDFData = G_InitialBRDFData(RampedLighting, Smoothness, Metallic, SpecularIntensity, IsEye);

    float3 IndirectSpecular = 0;
    float3 ReflectVector = reflect(-ViewDirection, NormalWS);
#ifdef _USE_REFLECTION_TEXTURE
		float ReflectionTextureMip = PerceptualRoughnessToMipmapLevel(G_BRDFData.perceptualRoughness);
        float3 VLSpecCube = texCUBE(sampler_VLSpecCube, ReflectVector);
        VLSpecCube *= _VLSpecColor;
        IndirectSpecular = VLSpecCube;
#endif
#ifdef _USE_EYE_REFLECTION_TEXTURE
		float ReflectionTextureMip = PerceptualRoughnessToMipmapLevel(G_BRDFData.perceptualRoughness);
        float3 VLSpecCube = texCUBE(sampler_VLSpecCube, ReflectVector);
        VLSpecCube *= _VLEyeSpecColor;
        IndirectSpecular = VLSpecCube;
#endif

    float3 MatCapReflection = 0.0f;
#ifdef _USE_REFLECTION_SPHERE
        float2 ReflectionSphereMapUV = NormalMatS.xy * 0.5 + 0.5;
        float4 ReflectionSphereMap = tex2D(sampler_ReflectionSphereMap, ReflectionSphereMapUV);
    
        float ReflectionSphereIntensity = lerp(1, ReflectionSphereMap.a, _ReflectionSphereMap_HDR.w);
        ReflectionSphereIntensity = max(ReflectionSphereIntensity, 0);
		ReflectionSphereIntensity = pow(ReflectionSphereIntensity, _ReflectionSphereMap_HDR.y);
        ReflectionSphereIntensity *= _ReflectionSphereMap_HDR.x;
    
        ReflectionSphereMap.xyz = ReflectionSphereMap.xyz * ReflectionSphereIntensity;
        MatCapReflection = ReflectionSphereMap.xyz;
#endif

    float FresnelTerm = Pow4(1 - saturate(NormalMatS.z)); // NormalMatS.z相当于NoV
    float3 SpecularColor = EnvironmentBRDFSpecular(G_BRDFData, FresnelTerm);
    float3 SpecularTerm = DirectBRDFSpecular(G_BRDFData, NormalWorM, _MatCapMainLight, ViewDirWorM);
    float3 Specular = SpecularColor * IndirectSpecular;
    Specular += SpecularTerm * SpecularColor;
    Specular += MatCapReflection;
    Specular *= SpecularIntensity;

    if (IsEyeBrow)
    {
        float2 EyeBrowHightLightUV = saturate(i.UV.xy + float2(-0.968750, -0.968750));
        float EyeBrowHightLightMask = EyeBrowHightLightUV.y * EyeBrowHightLightUV.x;
        EyeBrowHightLightMask = EyeBrowHightLightMask != 0.000000;
        Specular += EyeBrowHightLightMask ? RampedLighting * 2.0f : 0.0f;
    }
	
    Specular = lerp(Specular, Specular * RampAddColor, RampAddMap.w);

    float3 SH = SampleSH(NormalWS);
    float3 SkyLight = max(SH, 0) * _GlobalLightParameter.x * G_BRDFData.diffuse;

    float3 NormalVS = mul(UNITY_MATRIX_V, float4(NormalWS.xyz, 0.0)).xyz;
    float RimLight = 1 - dot(NormalVS, normalize(_MatCapRimLight.xyz));
    RimLight = pow(RimLight, _MatCapRimLight.w);
    float RimLightMask = min(DefDiffuse * DefDiffuse, 1.0f) * VertexColor.RimMask;
    RimLight = min(RimLight, 1.0f) * RimLightMask;

    float3 RimLightColor = lerp(1, RampedLighting, _MatCapRimColor.a) * _MatCapRimColor.xyz;
    RimLightColor *= RimLight;
	
    float3 OutLighting = G_BRDFData.diffuse;
    OutLighting += Specular;
    OutLighting *= _MatCapLightColor.xyz;
	
    float3 AdditionalLighting = 0;

    OutLighting += AdditionalLighting * _GlobalLightParameter.y;
    OutLighting += SkyLight;
    OutLighting += RimLightColor;
    OutLighting += RampMap.w * _ShadeAdditiveColor;

    OutLighting *= _MultiplyColor.xyz;
	
    float Alpha = BaseMap.a * _MultiplyColor.a;
#ifdef _ALPHATEST_ON
		clip(Alpha - _ClipValue);
#endif
#ifdef _ALPHAPREMULTIPLY_ON
        OutLighting *= Alpha;
#endif

    return float4(OutLighting, Alpha);
}

v2f vertOutline(appdata v)
{
    v2f o;
				
    float3 PositionWS = TransformObjectToWorld(v.Position);
    float3 SmoothNormalWS = TransformObjectToWorldNormal(v.Tangent);

    G_VertexColor VertexColor = DecodeVertexColor(v.Color);
    o.Color1 = VertexColor.OutLineColor;
    o.Color2 = float4(
					VertexColor.OutLineWidth,
					VertexColor.OutLineOffset,
					VertexColor.RampAddID,
					VertexColor.RimMask);

	
    float CameraDistance = length(_WorldSpaceCameraPos - PositionWS);
    float OutLineWidth = min(CameraDistance * _OutlineParam.z * _OutlineParam.w, 1.0f);
    OutLineWidth *= (_OutlineParam.y - _OutlineParam.x);
    OutLineWidth += (_OutlineParam.x);
    OutLineWidth *= 0.01f * VertexColor.OutLineWidth;
				
    float3 OffsetVector = OutLineWidth * SmoothNormalWS;
				
    float3 OffsetedPositionWS = PositionWS + OffsetVector;
				
    float4 OffsetedPositionCS = TransformWorldToHClip(OffsetedPositionWS);
    OffsetedPositionCS.z -= VertexColor.OutLineOffset * 6.66666747e-05;

    o.PositionCS = OffsetedPositionCS;
    o.Color1.xyz = VertexColor.OutLineColor;
				
    return o;
}

float4 fragOutline(v2f i, bool IsFront : SV_IsFrontFace) : SV_Target
{
    float3 OutLineColor = i.Color1.xyz * _MultiplyOutlineColor.xyz;
    float OutLineAlpha = _MultiplyColor.a;
    return float4(OutLineColor, OutLineAlpha);
}

technique MainTec < string MMDPass = "object"; >
{
    pass DarwObject
    {
        VertexShader = compile vs_3_0 vert();
        PixelShader = compile ps_3_0 frag();
        CULLMODE = CCW;
        ZENABLE = TRUE;
        ZWRITEENABLE = TRUE;
        ALPHABLENDENABLE = FALSE;
    }
    pass HairCover
    {
        VertexShader = compile vs_3_0 vert();
        PixelShader = compile ps_3_0 frag();
        CULLMODE = CCW;
        ZENABLE = TRUE;
        ZWRITEENABLE = FALSE;
        ALPHABLENDENABLE = TRUE;
        SRCBLEND = SRCALPHA;
        DESTBLEND = INVSRCALPHA;
    }
    pass DrawEdge
    {
        VertexShader = compile vs_3_0 vertOutline();
        PixelShader = compile ps_3_0 fragOutline();
        CULLMODE = CW;
        ZENABLE = TRUE;
        ZWRITEENABLE = TRUE;
        ALPHABLENDENABLE = FALSE;
    }
};

technique MainTec_ss < string MMDPass = "object_ss"; >
{
    pass DarwObject
    {
        VertexShader = compile vs_3_0 vert();
        PixelShader = compile ps_3_0 frag();
        CULLMODE = CCW;
        ZENABLE = TRUE;
        ZWRITEENABLE = TRUE;
        ALPHABLENDENABLE = FALSE;
    }
    pass HairCover
    {
        VertexShader = compile vs_3_0 vert();
        PixelShader = compile ps_3_0 frag();
        CULLMODE = CCW;
        ZENABLE = TRUE;
        ZWRITEENABLE = FALSE;
        ALPHABLENDENABLE = TRUE;
        SRCBLEND = SRCALPHA;
        DESTBLEND = INVSRCALPHA;
    }
    pass DrawEdge
    {
        VertexShader = compile vs_3_0 vertOutline();
        PixelShader = compile ps_3_0 fragOutline();
        CULLMODE = CW;
        ZENABLE = TRUE;
        ZWRITEENABLE = TRUE;
        ALPHABLENDENABLE = FALSE;
    }
};
