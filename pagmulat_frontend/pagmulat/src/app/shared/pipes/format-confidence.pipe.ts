import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'formatConfidence' })
export class FormatConfidencePipe implements PipeTransform {
  transform(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }
}
