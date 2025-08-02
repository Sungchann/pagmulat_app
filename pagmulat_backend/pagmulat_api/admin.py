from django.contrib import admin
from .models import AssociationRule

@admin.register(AssociationRule)
class AssociationRuleAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'support', 'confidence', 'lift')
    list_filter = ('confidence', 'lift')
    search_fields = ('antecedents', 'consequents')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Rule Definition', {
            'fields': ('antecedents', 'consequents')
        }),
        ('Metrics', {
            'fields': ('support', 'confidence', 'lift')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )