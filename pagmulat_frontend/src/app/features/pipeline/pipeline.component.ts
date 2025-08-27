import { Component, OnInit } from '@angular/core';
import { PipelineDataService } from 'src/app/services/pipeline-data.service';

@Component({
  selector: 'app-pipeline',
  templateUrl: './pipeline.component.html',
  styleUrls: ['./pipeline.component.css']
})
export class PipelineComponent implements OnInit {

  // Properly type raw and cleaned data
  rawData: { headers: string[]; rows: any[] } | null = null;
  cleanedData: { headers: string[]; rows: any[] } | null = null;
  augmentedData: { headers: string[]; rows: any[] } | null = null;

  constructor(private pipelineService: PipelineDataService) {}

  ngOnInit(): void {
    this.loadRawData();
    this.loadCleanedData();
    this.loadAugmentedData();
  }

  loadRawData(): void {
    this.pipelineService.getRawData().subscribe({
      next: (data) => {
        console.log('Raw Data:', data);
        this.rawData = data;
      },
      error: (err) => {
        console.error('Error loading raw data:', err);
      }
    });
  }

  loadCleanedData(): void {
    this.pipelineService.getCleanedData().subscribe({
      next: (data) => {
        console.log('Cleaned Data:', data);
        this.cleanedData = data;
      },
      error: (err) => {
        console.error('Error loading cleaned data:', err);
      }
    });
  }

  loadAugmentedData(): void{
    this.pipelineService.getAugmentedData().subscribe({
      next: (data) => {
        this.augmentedData = data
      },
      error: (err) => {
        console.error('Error loading augmented data:', err);
      }
    });
  }
}
