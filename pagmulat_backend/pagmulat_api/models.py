from django.db import models

class AssociationRule(models.Model):
    antecedents = models.JSONField(help_text="Left-hand side of the rule")
    consequents = models.JSONField(help_text="Right-hand side of the rule")
    support = models.FloatField()
    confidence = models.FloatField()
    lift = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        ant = ", ".join(self.antecedents)
        cons = ", ".join(self.consequents)
        return f"{ant} → {cons} (conf: {self.confidence:.2f})"
    
    class Meta:
        ordering = ['-confidence']
        verbose_name = "Behavior Association Rule"