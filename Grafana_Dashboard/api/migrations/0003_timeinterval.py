# Generated by Django 3.2.5 on 2021-07-30 08:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_alter_service_port'),
    ]

    operations = [
        migrations.CreateModel(
            name='TimeInterval',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time', models.IntegerField()),
            ],
        ),
    ]