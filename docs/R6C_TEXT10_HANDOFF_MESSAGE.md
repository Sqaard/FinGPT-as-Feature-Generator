# R6c Text10 Handoff Message

Короткое сообщение для чата:

```text
Проверили text-feature ablation для двух PPO-архитектур. На старой custom_custom текст ухудшает результат, но на новой R6c text10 дает небольшой плюс по return/Sharpe:

pair | return delta | Sharpe delta | max DD delta
custom_custom -> custom_custom&text10 | -12.31% | -0.7012 | -2.08%
R6c -> R6c&text10 | +0.31% | +0.0374 | -0.27%

Вывод: R6c лучше использует текстовые признаки, эту ветку можно брать как текущий baseline для следующих PPO text-feature прогонов.

Как запустить:
1. Склонировать ветку DRL: git clone -b DRL https://github.com/Sqaard/FinGPT-as-Feature-Generator.git
2. Взять launch package: artifacts/r6c_stage0_1_text_baseline_20260530/r6c_stage0_1_deepseek_v2_text_launch_package.zip
3. Распаковать его на Huawei/local machine.
4. Запустить smoke или full run:
   - run_r6c_text_smoke.ps1
   - run_r6c_text_full.ps1
```

Screenshot:

`artifacts/r6c_stage0_1_text_baseline_20260530/cross_model_text_effect_table.png`

