# Speed-up-Point-Feature-Histograms
Final project for ROB 422 in Umich

## Environment setting
```bash
conda env create -f environment.yml
conda activate pfh
```

If it is not working, try the following:
```bash
conda init cmd.exe
cmd /k "conda activate pfh"
```

## Github pull and push
```bash
git clone https://github.com/Pscjiayou/Speed-up-Point-Feature-Histograms.git
```

Getting the latest codes
```bash
git pull origin main
```

Creating branch
```bash
git checkout -b your_name-feature
```

Checking which branch it is
```bash
git branch
```

Pushing codes to your branch
```bash
git add .
git commit -m "add: 新增数据预处理模块"
git push origin your_name-feature
```
