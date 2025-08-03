// angular/src/app/services/api.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) { }

  /**
   * Fetches ARM metadata (top rules, stats, etc.)
   */
  getArmMetadata(): Observable<any> {
    return this.http.get(`${this.apiUrl}/arm/metadata`);
  }

  /**
   * Fetches ARM dashboard metrics (limited rules, fast)
   */
  getArmDashboard(): Observable<any> {
    return this.http.get(`${this.apiUrl}/arm/dashboard/`);
  }

  /**
   * Fetches all association rules
   */
  getAssociationRules(): Observable<any> {
    return this.http.get(`${this.apiUrl}/arm/rules`);
  }

  // getDashboard removed: ARM backend does not provide this endpoint anymore

  getBehaviorPatterns(behavior: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/patterns/${behavior}/`);
  }

  predictProductivity(features: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/predict/`, features);
  }

  trainModel(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train/`, {});
  }

  getAllRules(): Observable<any> {
    return this.http.get(`${this.apiUrl}/arm-rules/`);
  }

  getAllItemsets(): Observable<any> {
    return this.http.get(`${this.apiUrl}/frequent-itemsets/`);
  }

  getAllStudents(): Observable<any> {
    return this.http.get(`${this.apiUrl}/all_students/`);
  }

  getPredictionHistory(): Observable<any> {
    return this.http.get(`${this.apiUrl}/prediction-history/`);
  }
}