import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface Rule {
  antecedent: string;
  consequent: string;
  support: number;
  confidence: number;
  lift: number;
}

export interface AlgorithmRulesResponse {
  minSupport: number;
  minConfidence: number;
  minLift: number;
  aprioriRules: Rule[];
  fpGrowthRules: Rule[];
}

@Injectable({
  providedIn: 'root'
})
export class GeneratedRulesComparisonService {

  private apiUrl = 'http://127.0.0.1:8000/api/apriori-fpgrowth-rules/';

  constructor(private http: HttpClient) {}

  getRules(): Observable<AlgorithmRulesResponse> {
    return this.http.get<AlgorithmRulesResponse>(this.apiUrl);
  }
}
