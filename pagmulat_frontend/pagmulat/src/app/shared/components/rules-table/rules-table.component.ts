import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-rules-table',
  templateUrl: './rules-table.component.html',
  styleUrls: ['./rules-table.component.css']
})
export class RulesTableComponent {
  @Input() rules: any[] = [];
}
