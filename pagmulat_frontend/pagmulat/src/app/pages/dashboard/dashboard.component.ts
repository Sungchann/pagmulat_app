
import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  // Add these missing properties
  showSurveyModal = false;
  showDetail = false;
  detailType = '';
  isLoadingDetail = false;
  metrics: any = {};
  thresholds: any = {};
  rules: any[] = [];
  allStudents: any[] = [];
  allRules: any[] = [];
  allItemsets: any[] = [];
  predictionHistory: any[] = [];

  constructor(private apiService: ApiService) { }

  ngOnInit(): void {
    this.loadDashboard();
  }

  loadDashboard(): void {
    this.apiService.getDashboard().subscribe({
      next: (data) => {
        this.metrics = data.metrics;
        this.thresholds = data.thresholds;
        this.rules = data.rules_table;
      },
      error: (err) => console.error('Dashboard load failed', err)
    });
  }

  // Add these missing methods
  openSurveyModal(): void {
    this.showSurveyModal = true;
  }

  closeSurveyModal(): void {
    this.showSurveyModal = false;
  }

  openDetail(type: string): void {
    this.detailType = type;
    this.showDetail = true;
    this.isLoadingDetail = true;

    switch (type) {
      case 'students':
        this.apiService.getAllStudents().subscribe(students => {
          this.allStudents = students;
          this.isLoadingDetail = false;
        });
        break;
      case 'arm_rules':
        this.apiService.getAllRules().subscribe(rules => {
          this.allRules = rules;
          this.isLoadingDetail = false;
        });
        break;
      case 'itemsets':
        this.apiService.getAllItemsets().subscribe(itemsets => {
          this.allItemsets = itemsets;
          this.isLoadingDetail = false;
        });
        break;
      case 'prediction_history':
        this.apiService.getPredictionHistory().subscribe(history => {
          this.predictionHistory = history;
          this.isLoadingDetail = false;
        });
        break;
    }
  }

  startTraining(): void {
    this.apiService.trainModel().subscribe({
      next: (res) => console.log('Training started', res),
      error: (err) => console.error('Training failed', err)
    });
  }
}
