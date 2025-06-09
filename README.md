# GakuenMME
Gakuen Idolm@ster Shader for MMD  

![show](https://github.com/user-attachments/assets/6d618334-ad1f-4598-9ebd-f1952d0919cd)  
此MME基于 @Yu-ki016 的[shader逆向工程](https://github.com/Yu-ki016/Yu-ki016-Articles/tree/main/%E5%8D%A1%E9%80%9A%E6%B8%B2%E6%9F%93/%E5%AD%A6%E5%9B%AD%E5%81%B6%E5%83%8F%E5%A4%A7%E5%B8%88/%E8%A7%92%E8%89%B2%E6%B8%B2%E6%9F%93)改造  

# 使用方法
此MME需要使用顶点色数据才能正常工作  
推荐使用文章[《学院偶像大师资源提取记录》](https://croakfang.fun/2024/05/27/%e5%ad%a6%e9%99%a2%e5%81%b6%e5%83%8f%e5%a4%a7%e5%b8%88%e8%b5%84%e6%ba%90%e6%8f%90%e5%8f%96%e8%ae%b0%e5%bd%95/)的内附工程导出模型  
或者使用[UnityPMXExporter](https://github.com/croakfang/UnityPMXExporter)插件导出  

**1. 找到所有fx文件的如下内容，每个文件的内容可能会不同**
```
Texture2D _BaseMap : MATERIALTEXTURE;
Texture2D _ShadeMap < string ResourceName = "../Texture2D/t_chr_fktn-cstm-0000_bdyco_sdw.png"; >;
Texture2D _RampMap < string ResourceName = "../Texture2D/t_chr_fktn-base-0000_rmp.png"; >;
Texture2D _HighlightMap;
Texture2D _DefMap < string ResourceName = "../Texture2D/t_chr_fktn-cstm-0000_bdyco_def.png"; >;
Texture2D _LayerMap;
Texture2D _RampAddMap < string ResourceName = "../Texture2D/t_chr_fktn-cstm-0000_bdy_rma.png"; >;
Texture2D _ReflectionSphereMap;
textureCUBE _VLSpecCube < string ResourceName = "skybox.dds"; >;
```
除`skybox.dds`外，将有定义的贴图替换成你的模型的贴图  

**2. 将模型名字修改为英文，如：Kotone Fujita.pmx**  
**3. 模型添加一个骨骼`head`，将其父骨骼设为`頭`，并与`頭`位置重合**  
**4. 找到参数:**  
   ```
   float4x4 _HeadMatrix : CONTROLOBJECT < string name = "model.pmx"; string item = "head"; >;
   ```  
   将`name`替换为模型的名字，如：  
   ```
   float4x4 _HeadMatrix : CONTROLOBJECT < string name = "Kotone Fujita.pmx"; string item = "head"; >;
   ```

### 重要参数  
其它值不建议修改  
| 参数名            | 说明                 | 
|-------------------|---------------------|
|`SkinSaturation`   | 皮肤颜色饱和度        | 
|`_ClipValue`       | 贴图裁切，出现不明色块时尝试调整这个值|

| 参数名            | 说明                 | x   | y   | z   | w   |
|-------------------|---------------------|-----|-----|-----|-----|
|`_MainLightParam`  | 灯光参数             | /   | /   | /   | 打光类型，0为正面打光，1为环境灯光   |
| `_RampAddColor`   | 镭射高光效果颜色     | R  | G  | B | A  |
|`_MatCapRimColor`  | 边缘光颜色           | R  | G  | B  | 和原颜色的混合，为0时不混合  |
| `_MatCapRimLight` | 边缘光参数           | 方向X  | Y  | Z  | 边缘光范围，值越大范围越小  |
|`_ShadeMultiplyColor`| 阴影颜色          | R  | G  | B | A  |
|`_ShadeAdditiveColor`| 阴影叠加颜色      | R  | G  | B | A  |
|`_SpecularThreshold`|高光阈值            | 高光中心阈值  | 高光过渡宽度  | /| /  |
|`_OutlineParam`     | 描边参数           | 最小宽度 | 受距离影响的程度| 描边宽度 | /|
|`_MultiplyOutlineColor`| 描边乘加颜色     |  R  | G  | B | A  |
|`_EyeHighlightColor`|眼睛高光颜色         |  R  | G  | B | A  |

### 注意事项
- 当和其他FX一起使用时，头发可能会出现意外的穿透，此时将模型头发材质自带的贴图更改为无透明通道的贴图即可  
  例如头发模型使用：`t_chr_fktn-cstm-0000_hir_col_alp _no_alpha.png`  
  而头发FX使用：`Texture2D _BaseMap < string ResourceName = "../Texture2D/t_chr_fktn-cstm-0000_hir_col_alp.png"; >;`