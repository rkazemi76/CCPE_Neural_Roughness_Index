# مخزن محاسباتی مقاله شاخص زبری عصبی

این مخزن کدها، checkpointها، داده‌های مشتق‌شده و خروجی‌های محاسباتی پشتیبان نسخه 0.25 مقاله زیر را آرشیو می‌کند:

**A Simulation-to-Estimation Framework for a Neural Roughness Index in Cryptocurrency Volatility: Synthetic Validation, Exploratory Empirical Forecasting, and Computational Evaluation**

## ساختار

- `experiments/empirical/`: کد تحلیل پنج رمزارز، برآوردهای rolling، نتایج آماری و پیش‌بینی، و checkpointهای W384.
- `experiments/fixed_h/`: آزمایش مصنوعی H ثابت، ۹ checkpoint، scalerها، خروجی‌های کامل و اسکریپت بازتولید.
- `experiments/parallel_benchmark/`: کد و شواهد بنچمارک CPU با ۲، ۴ و ۶ worker.
- `experiments/smooth_semimartingale/`: کنترل OU برای فرایند نیمه‌مارتینگل هموار خارج از دامنه آموزش.
- `docs/`: راهنمای بازتولید، فرهنگ فایل‌ها و محدودیت‌های تفسیر.

قبل از اجرا، فایل `docs/REPRODUCIBILITY.md` و README هر آزمایش را مطالعه کنید. صحت فایل‌های منتشرشده را از ریشه مخزن با فرمان زیر بررسی کنید:

```bash
python scripts/verify_repository.py
```

## محدودیت‌های مهم

1. شاخص زبری عصبی در شبیه‌سازی‌های rough آموزش و ارزیابی شده است. خروجی رمزارزها یک شاخص توصیفی کالیبره‌شده با شبیه‌سازی است، نه حقیقت مشاهده‌شده زبری نهفته بازار.
2. کنترل OU یک آزمون تشخیصی خارج از دامنه است و classifier کالیبره‌شده rough/smooth یا آزمون فرض H=0.5 نیست.
3. نتایج forecasting به نمونه و مدل‌های بررسی‌شده محدودند؛ ادعای برتری همگانی مطرح نیست.
4. بنچمارک موازی فقط تولید مسیر مصنوعی، ساخت ویژگی‌های ۲۷ کاناله و انتقال خروجی را زمان‌سنجی می‌کند.
5. فایل بزرگ `features.npz` از آرشیو GitHub حذف شده است؛ هش نسخه اصلی آن در `source_run_manifest.json` نگهداری شده و اسکریپت Fixed-H امکان تولید مجدد آن را دارد.
6. داده خام یک‌دقیقه‌ای Binance در مخزن توزیع نشده؛ اسکریپت تجربی در صورت نیاز آن را از Binance Data Vision دریافت می‌کند.

**نکته مجوز:** هنوز فایل LICENSE اضافه نشده است. پیش از انتشار عمومی باید مجوز بازتوزیع و استفاده مجدد کد، مدل‌ها و خروجی‌ها توسط نویسندگان تعیین شود؛ جزئیات در `LICENSE_PENDING.md` آمده است.
