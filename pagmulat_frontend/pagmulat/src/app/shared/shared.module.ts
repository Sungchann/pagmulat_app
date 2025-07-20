import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FileUploaderComponent } from './components/file-uploader/file-uploader.component';
import { RulesTableComponent } from './components/rules-table/rules-table.component';
import { MetricsCardComponent } from './components/metrics-card/metrics-card.component';

@NgModule({
  declarations: [
    FileUploaderComponent,
    RulesTableComponent,
    MetricsCardComponent
  ],
  imports: [CommonModule],
  exports: [
    FileUploaderComponent,
    RulesTableComponent,
    MetricsCardComponent
  ]
})
export class SharedModule { }