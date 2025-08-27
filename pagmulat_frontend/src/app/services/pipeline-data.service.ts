import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PipelineDataService {

  private baseUrl = 'http://localhost:8000/api/';

  constructor(private http: HttpClient) {}

  getRawData(): Observable<any> {
    return this.http.get(`${this.baseUrl}raw-data/`);
  }

  getCleanedData(): Observable<any> {
    return this.http.get(`${this.baseUrl}cleaned-data/`);
  }

  getAugmentedData(): Observable<any> {
    return this.http.get(`${this.baseUrl}augmented-data/`);
  }
}
