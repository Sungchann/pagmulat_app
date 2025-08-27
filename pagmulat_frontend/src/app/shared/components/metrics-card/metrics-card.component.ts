import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-metrics-card',
  templateUrl: './metrics-card.component.html',
  styleUrls: ['./metrics-card.component.css']
})
export class MetricsCardComponent {
  @Input() metrics: any;
}
