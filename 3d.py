import pymol
from pymol import cmd
from pymol import util
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import tempfile


def create_from_smiles(smiles, object_name="mol"):
    """
    通过SMILES字符串创建分子（使用临时文件方法）
    """
    # 从SMILES创建分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")

    # 添加氢原子并生成3D坐标
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
        # 保存为SDF文件
        writer = Chem.SDWriter(tmp.name)
        writer.write(mol)
        writer.close()

        # 在PyMOL中加载
        cmd.load(tmp.name, object_name)

    # 删除临时文件
    os.unlink(tmp.name)


# 初始化PyMOL
pymol.finish_launching(['pymol', '-c'])

# 创建分子
create_from_smiles(r"CCOC(=O)/C=C/C")


# 设置显示和样式
cmd.show("sticks")
cmd.set("stick_radius", 0.12)
cmd.show("spheres", "all")
cmd.set("sphere_scale", 0.2)
cmd.set("sphere_scale", 0.1, "elem H")
# 使用util.cbag实现键的渐变着色
util.cbag("all")

# 设置原子颜色
cmd.color("grey", "elem C")  # 碳原子灰色
cmd.color("red", "elem O")   # 氧原子红色
cmd.color("blue", "elem N")  # 氮原子蓝色
cmd.color("white", "elem H") # 氢原子白色

# 调整视角
cmd.rotate("x", 0)  # 绕X轴旋转45度
cmd.rotate("y", 0)  # 绕Y轴旋转30度
cmd.zoom("all", 5)   # 放大视图
cmd.ray(1200, 1200)
cmd.png("111.png", dpi=300)