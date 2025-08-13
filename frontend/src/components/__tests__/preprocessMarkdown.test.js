import { describe, it, expect } from 'vitest';
import Message from '../Message';

// Extract preprocessMarkdown from Message.jsx for testing
function preprocessMarkdown(text, sources = [{}]) {
  if (!sources || sources.length === 0) return text;
  // Match all consecutive citation markers possibly followed by punctuation
  // Example: [2][3][4]. or [2][3], etc.
  return text.replace(/((\[\d+\])+)([.,;:!?])?/g, (match, markers, _, punct) => {
    // Split markers: [2][3][4] => ['[2]', '[3]', '[4]']
    const markerArr = markers.match(/\[\d+\]/g) || [];
    // Convert each marker to {{CITATION_n}}
    const converted = markerArr.map(m => {
      const n = m.match(/\d+/)[0];
      return `{{CITATION_${n}}}`;
    }).join(' ');
    // Add punctuation if present
    return punct ? `${converted}${punct}` : converted;
  });
}

describe('preprocessMarkdown', () => {
  it('should convert consecutive citations and punctuation correctly', () => {
    const input = `A CIF ... for the fund's investors [3].\n\n...Open-ended funds ... [5].\n...disclosures to investors [3].\n...investment strategies ... [5].\n...professional management ... [3].\n...entities ... [3].\n...types of CIFs ... [5].`;
    const expected = `A CIF ... for the fund's investors {{CITATION_3}}.\n\n...Open-ended funds ... {{CITATION_5}}.\n...disclosures to investors {{CITATION_3}}.\n...investment strategies ... {{CITATION_5}}.\n...professional management ... {{CITATION_3}}.\n...entities ... {{CITATION_3}}.\n...types of CIFs ... {{CITATION_5}}.`;
    expect(preprocessMarkdown(input, [{}])).toBe(expected);
  });

  it('should handle multiple consecutive citations and punctuation', () => {
    const input = '...requirements [3][5].';
    const expected = '...requirements {{CITATION_3}} {{CITATION_5}}.';
    expect(preprocessMarkdown(input, [{}])).toBe(expected);
  });

  it('should handle citations with no punctuation', () => {
    const input = '...investors [3][5]';
    const expected = '...investors {{CITATION_3}} {{CITATION_5}}';
    expect(preprocessMarkdown(input, [{}])).toBe(expected);
  });

  it('should handle citations with comma', () => {
    const input = '...investors [3][5], next.';
    const expected = '...investors {{CITATION_3}} {{CITATION_5}}, next.';
    expect(preprocessMarkdown(input, [{}])).toBe(expected);
  });
});
