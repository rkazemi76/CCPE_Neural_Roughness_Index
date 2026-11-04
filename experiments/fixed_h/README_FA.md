# آزمایش Fixed-H — بسته بازتولیدپذیری

این آزمایش یک تشخیص مصنوعی جدید و کنترل‌شده است؛ نه بازتولید تاریخی، بازآموزی مدل‌های گزارش‌شده در مقاله، یا اعتبارسنجی H واقعی بازار. جزئیات طراحی، شرایط آزمایش و محدودیت تفسیر در `PROTOCOL.md` آمده است.

## پیش‌نیاز و اجرا

از پوشه `experiments/fixed_h/` اجرا کنید. محیط پیشنهادی Python 3.11 و PyTorch واقعی روی CPU است:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python run_experiment.py --threads 1
```

برای آزمون مهندسی کوتاه:

```bash
python run_experiment.py --smoke-only --threads 1
```

هر اجرا یک پوشه جدید با زمان UTC در `runs/` می‌سازد. ابتدا کدها و ۴۱ ورودی منبع بررسی می‌شوند؛ سپس smoke عصبی واقعی اجرا می‌شود و فقط در صورت PASS، اجرای full آغاز می‌شود. از `-O` یا `PYTHONOPTIMIZE` استفاده نکنید.

## فایل ویژگی حجیم

آرایه `features.npz` با حجم تقریبی ۷۰۰ مگابایت از این مخزن GitHub حذف شده تا از سقف اندازه فایل تکی GitHub عبور نکند. هش نسخه اصلی آن در `results/reported_full/source_run_manifest.json` حفظ شده است. سایر فایل‌های اجرای full و تمام checkpointها، scalerها و کدهای لازم در مخزن حضور دارند. اجرای full با `run_experiment.py` آرایه ویژگی را در پوشه محلی `runs/<timestamp>/full/` مجدداً تولید می‌کند؛ فایل را در ZIP بازگشت قرار نمی‌دهد.

## حدود نتیجه

- سه روش `single_raw`, `single_derived`, و `multi_original` با سه seed اصلی (42، 123، 2024) ارزیابی می‌شوند.
- شش وضعیت کنترل و دو سطح نویز مطابق `PROTOCOL.md` هستند.
- خروجی‌ها به طراحی مصنوعی حاضر و مدل‌های ثابت محدودند. نتایج، شاهدی برای H واقعی رمزارزها یا تعمیم به همه رژیم‌های بازار نیستند.
