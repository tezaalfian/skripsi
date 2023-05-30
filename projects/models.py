from django.db import models

class Claim(models.Model):
    date = models.DateField()
    amount = models.IntegerField()