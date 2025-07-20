// core/services/api.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = 'http://your-api-url/api';

  constructor(private http: HttpClient) { }

  uploadStudentLogs(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/upload`, formData);
  }

  runARM(params: any) {
    return this.http.post(`${this.baseUrl}/analyze`, params);
  }
}