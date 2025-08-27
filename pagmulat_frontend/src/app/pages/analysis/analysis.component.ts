import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { AssociationRule, ArmMetadata } from '../../shared/models/association-rule.model';

@Component({
  selector: 'app-analysis',
  templateUrl: './analysis.component.html',
  styleUrls: ['./analysis.component.css']
})
export class AnalysisComponent implements OnInit {
  metadata: ArmMetadata | null = null;
  rules: AssociationRule[] = [];
  loading = false;
  error: string | null = null;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchMetadata();
    this.fetchRules();
  }

  fetchMetadata() {
    this.loading = true;
    this.api.getArmMetadata().subscribe({
      next: (data: any) => {
        this.metadata = data;
        this.loading = false;
      },
      error: err => {
        this.error = 'Failed to load ARM metadata.';
        this.loading = false;
      }
    });
  }

  fetchRules() {
    this.api.getAssociationRules().subscribe({
      next: (data: any) => {
        this.rules = data;
      },
      error: err => {
        this.error = 'Failed to load association rules.';
      }
    });
  }
}
